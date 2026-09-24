"""跨事件概率校准报告测试：POST /events/calibration-report。

SQLite 文件库 + 模拟时钟，覆盖：
  - 1～100 个事件 ID：空列表 / 101 个返回 422；重复 ID 去重且保序
  - 任一事件不存在 404；未结算 409 CALIBRATION_EVENT_NOT_SETTLED；
    结算标准结果不是精确 "0"/"1"（" 1"、"1.0"、"yes" 等）422
    CALIBRATION_STANDARD_RESULT_NOT_BINARY；未结算优先于结果格式
  - 概率解析：无空白 / 无符号 / 无指数且位于 [0,1] 的纯十进制才有效；
    无效原文逐条列出（INVALID_FORMAT / OUT_OF_RANGE）；未揭示承诺单独计数
  - Decimal 精确计算：逐预测平方误差、总体 Brier 均分、十区间均值与
    实际为 1 频率；边界归属（0.1 属 [0.1,0.2)，1 属 [0.9,1]）；空样本 null
  - 四舍六入五成双至 6 位小数；事件与明细固定顺序；纯只读
  - 纯函数模块独立用例
"""

import os
import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import UUID, uuid4

_tmpdir = tempfile.mkdtemp(prefix="sealed_predictions_calibration_test_")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmpdir}/calibration.db"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app.clock as clock_module  # noqa: E402
import app.main as main_module  # noqa: E402
from app import calibration  # noqa: E402
from app.hashing import compute_commitment  # noqa: E402

T0 = datetime(2030, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
COMMIT_DL = T0 + timedelta(hours=1)
REVEAL_DL = T0 + timedelta(hours=2)


class MutableClock:
    def __init__(self, start: datetime):
        self.t = start

    def __call__(self) -> datetime:
        return self.t


clock = MutableClock(T0)

client = TestClient(main_module.app)


@pytest.fixture(autouse=True)
def _fixed_clock(monkeypatch):
    clock.t = T0
    monkeypatch.setattr(clock_module, "now_utc", clock)
    with client:  # 触发 lifespan 建表（幂等）
        yield


def create_event() -> str:
    clock.t = T0  # 截止时间固定在 T0 之后，创建时必须处于 commit 阶段
    r = client.post("/events", json={
        "title": "概率校准",
        "commit_deadline": COMMIT_DL.isoformat(),
        "reveal_deadline": REVEAL_DL.isoformat(),
    })
    assert r.status_code == 201, r.text
    return r.json()["id"]


def commit(event_id: str, text: str, salt: str) -> str:
    h = compute_commitment(text, salt)
    r = client.post(f"/events/{event_id}/commitments", json={"commitment_hash": h})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def commit_only(event_id: str, text: str, salt: str) -> None:
    """提交承诺但永不揭示（计入未揭示）。"""
    commit(event_id, text, salt)


def reveal(event_id: str, text: str, salt: str) -> None:
    r = client.post(
        f"/events/{event_id}/reveals",
        json={"original_text": text, "salt": salt},
    )
    assert r.status_code == 201, r.text


def settle(event_id: str, standard_result: str, expected: int = 200) -> dict:
    r = client.post(
        f"/events/{event_id}/settlement",
        json={"standard_result": standard_result},
    )
    assert r.status_code == expected, r.text
    return r.json()


def report(event_ids, expected: int = 200) -> dict:
    r = client.post(
        "/events/calibration-report",
        json={"event_ids": [str(x) for x in event_ids]},
    )
    assert r.status_code == expected, r.text
    return r.json()


def settle_event_with_predictions(
    standard_result: str,
    revealed_texts: list[tuple[str, str]],
    unrevealed_texts: list[tuple[str, str]] | None = None,
) -> tuple[str, list[str]]:
    """创建一个事件，提交并（按给定时刻间隔）揭示预测，closed 后结算。

    返回 (event_id, 承诺 ID 列表[与 revealed_texts 同序])。
    """
    clock.t = T0
    event_id = create_event()
    commitment_ids = []
    for i, (text, salt) in enumerate(revealed_texts):
        # 逐条错开提交时刻（commit 阶段内），随后在 reveal 阶段逐条揭示
        clock.t = T0 + timedelta(seconds=i)
        commitment_ids.append(commit(event_id, text, salt))
        clock.t = COMMIT_DL + timedelta(seconds=i)
        reveal(event_id, text, salt)
    for j, (text, salt) in enumerate(unrevealed_texts or []):
        # 未揭示承诺排在已揭示承诺之后提交（时刻唯一且仍在 commit 阶段）
        clock.t = T0 + timedelta(seconds=len(revealed_texts) + j)
        commit_only(event_id, text, salt)
    clock.t = REVEAL_DL
    settle(event_id, standard_result)
    clock.t = T0
    return event_id, commitment_ids


# --------------------------------------------------------------------------- #
# 请求体校验
# --------------------------------------------------------------------------- #

def test_empty_event_ids_is_422():
    r = client.post("/events/calibration-report", json={"event_ids": []})
    assert r.status_code == 422


def test_more_than_100_event_ids_is_422():
    ids = [str(uuid4()) for _ in range(101)]
    r = client.post("/events/calibration-report", json={"event_ids": ids})
    assert r.status_code == 422


def test_non_uuid_event_id_is_422():
    r = client.post(
        "/events/calibration-report", json={"event_ids": ["not-a-uuid"]}
    )
    assert r.status_code == 422


def test_missing_event_ids_field_is_422():
    r = client.post("/events/calibration-report", json={})
    assert r.status_code == 422


# --------------------------------------------------------------------------- #
# 前置业务校验
# --------------------------------------------------------------------------- #

def test_missing_event_is_404():
    body = report(["00000000-0000-0000-0000-000000000000"], expected=404)
    assert body["detail"]["code"] == "EVENT_NOT_FOUND"


def test_duplicate_ids_dedup_and_preserve_order():
    e1, _ = settle_event_with_predictions("1", [("1", "s1")])
    e2, _ = settle_event_with_predictions("0", [("0", "s2")])
    body = report([e1, e2, e1, e2])
    assert body["event_count"] == 2
    assert body["event_ids"] == [e1, e2]


def test_unsettled_event_is_409():
    event_id = create_event()  # commit 阶段未结算
    body = report([event_id], expected=409)
    assert body["detail"]["code"] == "CALIBRATION_EVENT_NOT_SETTLED"

    # closed 但未结算同样拒绝
    clock.t = REVEAL_DL
    body = report([event_id], expected=409)
    assert body["detail"]["code"] == "CALIBRATION_EVENT_NOT_SETTLED"


def test_non_binary_standard_result_is_422():
    # 注意：空串无法通过结算请求校验（无法写入结算记录），不在本接口处理范围
    for bad in [" 1", "1 ", "1.0", "00", "0.0", "yes", "no", "true", "２"]:
        clock.t = T0
        event_id = create_event()
        clock.t = REVEAL_DL
        settle(event_id, bad)
        body = report([event_id], expected=422)
        assert body["detail"]["code"] == "CALIBRATION_STANDARD_RESULT_NOT_BINARY", bad
        clock.t = T0


def test_unsettled_check_takes_precedence_over_non_binary():
    # 请求序中先遇到未结算事件即报 409，即使后面的事件结果非 0/1
    unsettled = create_event()  # commit 阶段，未结算
    bad = create_event()
    clock.t = REVEAL_DL
    settle(bad, "yes")
    body = report([unsettled, bad], expected=409)
    assert body["detail"]["code"] == "CALIBRATION_EVENT_NOT_SETTLED"
    # 换成已结算的非二值事件则暴露 422
    clock.t = T0
    other, _ = settle_event_with_predictions("0", [("0", "s")])
    body = report([other, bad], expected=422)
    assert body["detail"]["code"] == "CALIBRATION_STANDARD_RESULT_NOT_BINARY"


def test_missing_event_takes_precedence_over_unsettled():
    unsettled = create_event()
    missing = "00000000-0000-0000-0000-000000000000"
    body = report([unsettled, missing], expected=404)
    assert body["detail"]["code"] == "EVENT_NOT_FOUND"


# --------------------------------------------------------------------------- #
# 报告主流程
# --------------------------------------------------------------------------- #

def test_full_report_brier_bins_and_ordering():
    # 事件 A 实际为 1：
    #   0.8  -> bin 8, 误差 0.04
    #   0.6  -> bin 6, 误差 0.16
    #   1    -> bin 9, 误差 0
    #   1.2  -> OUT_OF_RANGE
    #   0.9  -> bin 9, 误差 0.01
    #   "概率" -> INVALID_FORMAT
    #   0.55 未揭示
    a_texts = [("0.8", "a1"), ("0.6", "a2"), ("1", "a3"),
               ("1.2", "a4"), ("0.9", "a5"), ("概率", "a6")]
    a, a_cids = settle_event_with_predictions("1", a_texts, [("0.55", "a7")])

    # 事件 B 实际为 0：
    #   0.1 -> bin 1, 误差 0.01
    #   0   -> bin 0, 误差 0
    #   .5  -> bin 5, 误差 0.25
    #   +1  -> INVALID_FORMAT
    b_texts = [("0.1", "b1"), ("0", "b2"), (".5", "b3"), ("+1", "b4")]
    b, b_cids = settle_event_with_predictions("0", b_texts)

    body = report([a, b])
    assert body["event_count"] == 2
    assert body["event_ids"] == [a, b]
    assert body["revealed_count"] == 10
    assert body["unrevealed_commitment_count"] == 1
    assert body["valid_prediction_count"] == 7
    assert body["invalid_prediction_count"] == 3

    # Brier 均分（精确误差上平均后舍入）：
    # (0.04 + 0.16 + 0 + 0.01 + 0.01 + 0 + 0.25) / 7 = 0.47 / 7
    assert body["brier_mean"] == str(
        (Decimal("0.47") / Decimal("7")).quantize(
            Decimal("0.000001"), rounding="ROUND_HALF_EVEN"
        )
    )

    # 区间固定为 10 个，下标连续
    bins = body["bins"]
    assert len(bins) == 10
    assert [bi["index"] for bi in bins] == list(range(10))
    assert bins[0]["lower_inclusive"] == "0.000000"
    assert bins[0]["upper_exclusive"] == "0.100000"
    assert bins[9]["lower_inclusive"] == "0.900000"
    assert bins[9]["upper_exclusive"] == "1.000000"

    by_index = {bi["index"]: bi for bi in bins}
    # bin 0: {0} 实际 0 -> 均值 0，频率 0
    assert by_index[0]["sample_count"] == 1
    assert by_index[0]["mean_prediction"] == "0.000000"
    assert by_index[0]["positive_rate"] == "0.000000"
    # bin 1: {0.1} 实际 0
    assert by_index[1]["sample_count"] == 1
    assert by_index[1]["mean_prediction"] == "0.100000"
    assert by_index[1]["positive_rate"] == "0.000000"
    # bin 5: {0.5} 实际 0
    assert by_index[5]["sample_count"] == 1
    assert by_index[5]["mean_prediction"] == "0.500000"
    # bin 6: {0.6} 实际 1 -> 频率 1
    assert by_index[6]["sample_count"] == 1
    assert by_index[6]["mean_prediction"] == "0.600000"
    assert by_index[6]["positive_rate"] == "1.000000"
    # bin 8: {0.8} 实际 1
    assert by_index[8]["sample_count"] == 1
    # bin 9: {1, 0.9} 实际均为 1 -> 均值 0.95，频率 1
    assert by_index[9]["sample_count"] == 2
    assert by_index[9]["mean_prediction"] == "0.950000"
    assert by_index[9]["positive_rate"] == "1.000000"
    # 空样本区间均值 / 频率为 null
    for i in [2, 3, 4, 7]:
        assert by_index[i]["sample_count"] == 0
        assert by_index[i]["mean_prediction"] is None
        assert by_index[i]["positive_rate"] is None

    # 有效明细：先按事件请求序（A 后 B），事件内按承诺提交时间
    valids = body["valid_predictions"]
    assert [p["event_id"] for p in valids] == [a] * 4 + [b] * 3
    assert [p["commitment_id"] for p in valids[:4]] == a_cids[:3] + a_cids[4:5]
    assert [p["commitment_id"] for p in valids[4:]] == b_cids[:3]

    # 实际结果逐预测回显
    assert {p["actual"] for p in valids if p["event_id"] == a} == {1}
    assert {p["actual"] for p in valids if p["event_id"] == b} == {0}

    # 逐预测平方误差
    se = {(p["original_text"], p["squared_error"]) for p in valids}
    assert se == {
        ("0.8", "0.040000"),
        ("0.6", "0.160000"),
        ("1", "0.000000"),
        ("0.9", "0.010000"),
        ("0.1", "0.010000"),
        ("0", "0.000000"),
        (".5", "0.250000"),
    }
    # 概率输出统一 6 位小数（原文另在 original_text 逐字回显）
    assert next(p for p in valids if p["original_text"] == "1")["probability"] == "1.000000"

    # 无效原文逐条列出并注明原因
    invalids = body["invalid_predictions"]
    assert {(i["original_text"], i["reason"]) for i in invalids} == {
        ("1.2", "OUT_OF_RANGE"),
        ("概率", "INVALID_FORMAT"),
        ("+1", "INVALID_FORMAT"),
    }
    # 无效明细同样按事件序、提交时间排序（A 的两条在 B 之前）
    assert [i["event_id"] for i in invalids] == [a, a, b]
    assert all(i["commitment_id"] for i in invalids)


def test_zero_and_one_probabilities_are_valid_and_boundary_bins():
    """边界归属：0 属 bin0；0.1 不属 bin0；1 属 bin9（而非不存在的 bin10）。"""
    e, _ = settle_event_with_predictions("0", [
        ("0", "s0"), ("0.099999", "s1"), ("0.1", "s2"), ("1", "s3"),
    ])
    body = report([e])
    counts = {bi["index"]: bi["sample_count"] for bi in body["bins"]}
    assert counts[0] == 2
    assert counts[1] == 1
    assert counts[9] == 1
    # Brier = (0^2 + 0.099999^2 + 0.1^2 + 1^2) / 4，精确十进制
    expected = (
        Decimal("0.099999") ** 2 + Decimal("0.1") ** 2 + Decimal("1") ** 2
    ) / Decimal(4)
    assert body["brier_mean"] == str(
        expected.quantize(Decimal("0.000001"), rounding="ROUND_HALF_EVEN")
    )


def test_invalid_probability_variants():
    bad_texts = [
        " 0.5",     # 前导空白
        "0.5 ",     # 尾随空白
        "0.5\n",    # 换行
        "+0.5",     # 正号
        "-0",       # 负号
        "1e-1",     # 指数
        ".5E0",     # 指数
        "0x1",      # 非十进制
        "0.5.6",    # 多个小数点
        "1_000",    # 下划线
        ".",        # 仅小数点
        "",         # 空原文
        "NaN",      # 非数字
        "Infinity",
        "∞",
        "0/1",      # 分数形式
    ]
    out_of_range = ["1.0000001", "2", "100", "1.1"]
    all_texts = bad_texts + out_of_range
    e, _ = settle_event_with_predictions("1", [(t, f"s-{i}") for i, t in enumerate(all_texts)])
    body = report([e])
    assert body["valid_prediction_count"] == 0
    assert body["invalid_prediction_count"] == len(all_texts)
    assert body["brier_mean"] is None  # 无有效样本
    reasons = {i["original_text"]: i["reason"] for i in body["invalid_predictions"]}
    for t in bad_texts:
        assert reasons[t] == "INVALID_FORMAT", t
    for t in out_of_range:
        assert reasons[t] == "OUT_OF_RANGE", t
    # 所有区间空样本
    assert all(bi["sample_count"] == 0 for bi in body["bins"])
    assert all(bi["mean_prediction"] is None for bi in body["bins"])
    assert all(bi["positive_rate"] is None for bi in body["bins"])


def test_valid_probability_formats():
    good = ["0", "1", "0.0", "1.0", ".0", ".5", "0.", "000.500", "0.999999"]
    e, cids = settle_event_with_predictions("1", [(t, f"s-{i}") for i, t in enumerate(good)])
    body = report([e])
    assert body["invalid_prediction_count"] == 0
    assert body["valid_prediction_count"] == len(good)
    by_text = {p["original_text"]: p for p in body["valid_predictions"]}
    # 概率输出统一四舍六入五成双至 6 位小数；原文逐字保留在 original_text
    assert by_text["1"]["probability"] == "1.000000"
    assert by_text[".5"]["probability"] == "0.500000"
    assert by_text["000.500"]["probability"] == "0.500000"
    assert by_text["0.999999"]["probability"] == "0.999999"
    assert by_text["0."]["squared_error"] == "1.000000"  # 概率 0，实际 1


def test_bin_mean_and_positive_rate():
    # bin 0 三个预测：实际依次 1, 0, 0 -> 频率 1/3，均值 (0+0.05+0.02)/3
    # 实际结果是事件级字段，用一个结果为 1 的事件加两个结果为 0 的事件组合
    e1, _ = settle_event_with_predictions("1", [("0", "k1")])
    e2, _ = settle_event_with_predictions("0", [("0.05", "k2"), ("0.02", "k3")])
    body = report([e1, e2])
    bin0 = body["bins"][0]
    assert bin0["sample_count"] == 3
    assert bin0["positive_rate"] == str(
        (Decimal(1) / Decimal(3)).quantize(
            Decimal("0.000001"), rounding="ROUND_HALF_EVEN"
        )
    )
    assert bin0["mean_prediction"] == str(
        (Decimal("0.07") / Decimal(3)).quantize(
            Decimal("0.000001"), rounding="ROUND_HALF_EVEN"
        )
    )


def test_half_even_rounding_to_six_places():
    # 预测 0.1234567、实际 1：误差 = 0.8765433^2 = 0.768328...
    e, _ = settle_event_with_predictions("1", [("0.1234567", "k")])
    body = report([e])
    expected = (Decimal("1") - Decimal("0.1234567")) ** 2
    assert body["valid_predictions"][0]["squared_error"] == str(
        expected.quantize(Decimal("0.000001"), rounding="ROUND_HALF_EVEN")
    )
    # 概率本身也统一 6 位小数（0.1234567 -> 0.123457），计算仍用精确值
    assert body["valid_predictions"][0]["probability"] == "0.123457"
    assert body["brier_mean"] == str(
        expected.quantize(Decimal("0.000001"), rounding="ROUND_HALF_EVEN")
    )

    # 构造五成双边界：三个预测均值恰为 x.xxxxxx5 不现实，直接对纯函数验证
    q = calibration.quantize_six
    assert q(Decimal("0.1234565")) == Decimal("0.123456")  # 前位偶，舍
    assert q(Decimal("0.1234555")) == Decimal("0.123456")  # 前位奇，入
    assert q(Decimal("0.1234575")) == Decimal("0.123458")


def test_report_is_read_only_and_deterministic():
    e1, _ = settle_event_with_predictions("1", [("0.8", "a"), ("bad", "b")])
    e2, _ = settle_event_with_predictions("0", [("0.1", "c")], [("0.3", "d")])

    first = report([e1, e2])
    second = report([e1, e2])
    assert first == second  # 冻结时钟下逐字节一致

    # 结算结果不受影响
    r = client.get(f"/events/{e1}/settlement")
    assert r.json()["status"] == "settled"
    assert r.json()["standard_result"] == "1"

    # 承诺 / 揭示计数不变
    r = client.get(f"/events/{e2}")
    assert r.json()["commitment_count"] == 2
    assert r.json()["reveal_count"] == 1


def test_zero_commitments_events_and_no_valid_predictions():
    # 两个零承诺事件均结算为 0/1
    e1 = create_event()
    e2 = create_event()
    clock.t = REVEAL_DL
    settle(e1, "0")
    settle(e2, "1")
    body = report([e1, e2])
    assert body["revealed_count"] == 0
    assert body["unrevealed_commitment_count"] == 0
    assert body["valid_prediction_count"] == 0
    assert body["invalid_prediction_count"] == 0
    assert body["brier_mean"] is None
    assert len(body["bins"]) == 10
    assert all(bi["sample_count"] == 0 for bi in body["bins"])
    assert body["valid_predictions"] == []
    assert body["invalid_predictions"] == []


def test_request_order_controls_event_and_detail_order():
    # 用可区分的揭示时刻保证事件内顺序固定；跨事件顺序由请求序决定
    e1, cids1 = settle_event_with_predictions("1", [("0.7", "x"), ("0.2", "y")])
    e2, cids2 = settle_event_with_predictions("0", [("0.4", "z")])

    body_ab = report([e1, e2])
    assert [p["event_id"] for p in body_ab["valid_predictions"]] == [e1, e1, e2]
    body_ba = report([e2, e1])
    assert body_ba["event_ids"] == [e2, e1]
    assert [p["event_id"] for p in body_ba["valid_predictions"]] == [e2, e1, e1]
    # 汇总数值与事件顺序无关
    assert body_ab["brier_mean"] == body_ba["brier_mean"]
    assert body_ab["bins"] == body_ba["bins"]


def test_accepts_100_settled_events():
    ids = []
    for i in range(100):
        clock.t = T0
        e, _ = settle_event_with_predictions("1" if i % 2 else "0", [(f"0.{(i % 9) + 1}", f"s{i}")])
        ids.append(e)
    body = report(ids)
    assert body["event_count"] == 100
    assert body["valid_prediction_count"] == 100


# --------------------------------------------------------------------------- #
# 纯函数模块
# --------------------------------------------------------------------------- #

def test_parse_probability_strict():
    ok = {"0": Decimal("0"), "1": Decimal("1"), "0.5": Decimal("0.5"),
          ".5": Decimal(".5"), "1.": Decimal("1"), "000.100": Decimal("000.100"),
          "0.999999": Decimal("0.999999")}
    for text, value in ok.items():
        parsed, reason = calibration.parse_probability(text)
        assert reason is None and parsed == value, text

    format_bad = ["", ".", " 0.5", "0.5 ", "+1", "-0", "1e0", ".5e-1",
                  "nan", "NaN", "Infinity", "1_000", "0x1", "1.2.3", "＋１"]
    for text in format_bad:
        parsed, reason = calibration.parse_probability(text)
        assert parsed is None and reason == calibration.INVALID_FORMAT, text

    range_bad = ["1.0000001", "2", "1.1", "999"]
    for text in range_bad:
        parsed, reason = calibration.parse_probability(text)
        assert parsed is None and reason == calibration.OUT_OF_RANGE, text


def test_bin_index_boundaries():
    f = calibration.bin_index_for
    assert f(Decimal("0")) == 0
    assert f(Decimal("0.099999")) == 0
    assert f(Decimal("0.1")) == 1
    assert f(Decimal("0.2")) == 2
    assert f(Decimal("0.899999")) == 8
    assert f(Decimal("0.9")) == 9
    assert f(Decimal("1")) == 9


def _uuid(i: int) -> UUID:
    return UUID(int=i)


def test_build_report_pure_brier_and_bins():
    outcomes = [(_uuid(1), 1), (_uuid(2), 0)]
    base = datetime(2030, 1, 1, tzinfo=timezone.utc)
    revealed = [
        # (event, commitment, committed_at, text)
        (_uuid(2), _uuid(30), base, "0.1"),    # bin1, 实际0, err 0.01
        (_uuid(1), _uuid(10), base, "0.8"),    # bin8, 实际1, err 0.04
        (_uuid(1), _uuid(11), base, "1.2"),    # OUT_OF_RANGE
        (_uuid(1), _uuid(12), base, "x"),      # INVALID_FORMAT
        (_uuid(2), _uuid(31), base, "0.4"),    # bin4, 实际0, err 0.16
    ]
    result = calibration.build_calibration_report(outcomes, revealed, unrevealed_count=2)

    assert result.unrevealed_count == 2
    assert result.valid_count == 3
    assert result.invalid_count == 2
    # 事件 1 的明细在事件 2 之前（请求序），事件内按承诺 ID
    assert [p.commitment_id for p in result.valid_predictions] == [
        _uuid(10), _uuid(30), _uuid(31)
    ]
    assert [p.event_id for p in result.valid_predictions] == [
        _uuid(1), _uuid(2), _uuid(2)
    ]
    assert [i.commitment_id for i in result.invalid_predictions] == [
        _uuid(11), _uuid(12)
    ]
    # Brier = (0.04 + 0.01 + 0.16) / 3 = 0.07
    assert result.brier_mean == Decimal("0.070000")
    # 平方误差按条量化
    se_by_cid = {p.commitment_id: p.squared_error for p in result.valid_predictions}
    assert se_by_cid[_uuid(10)] == Decimal("0.040000")
    assert se_by_cid[_uuid(30)] == Decimal("0.010000")
    assert se_by_cid[_uuid(31)] == Decimal("0.160000")

    bins = result.bins
    assert len(bins) == 10
    assert bins[1].sample_count == 1
    assert bins[1].mean_prediction == Decimal("0.100000")
    assert bins[1].positive_rate == Decimal("0.000000")
    assert bins[4].sample_count == 1
    assert bins[4].positive_rate == Decimal("0.000000")
    assert bins[8].sample_count == 1
    assert bins[8].positive_rate == Decimal("1.000000")
    assert bins[8].mean_prediction == Decimal("0.800000")
    assert bins[0].sample_count == 0
    assert bins[0].mean_prediction is None
    assert bins[0].positive_rate is None


def test_build_report_empty_valid_predictions_brier_is_none():
    result = calibration.build_calibration_report(
        [(_uuid(1), 0)],
        [(_uuid(1), _uuid(2), datetime(2030, 1, 1, tzinfo=timezone.utc), "nope")],
        unrevealed_count=0,
    )
    assert result.valid_count == 0
    assert result.brier_mean is None
    assert all(b.sample_count == 0 for b in result.bins)


def test_bin_mean_uses_exact_decimal_not_float():
    # bin 5 中两个预测 0.33 与 0.34 实际均为 1：均值 0.335，五成双至 0.335000
    base = datetime(2030, 1, 1, tzinfo=timezone.utc)
    revealed = [
        (_uuid(1), _uuid(1), base, "0.30"),  # bin3
        (_uuid(1), _uuid(2), base, "0.35"),  # bin3
        (_uuid(1), _uuid(3), base, "0.33"),  # bin3（全部落 bin3: [0.3,0.4)）
    ]
    result = calibration.build_calibration_report(
        [(_uuid(1), 1)], revealed, unrevealed_count=0
    )
    bin3 = result.bins[3]
    assert bin3.sample_count == 3
    # (0.30 + 0.35 + 0.33) / 3 = 0.98/3
    assert bin3.mean_prediction == (Decimal("0.98") / Decimal(3)).quantize(
        Decimal("0.000001"), rounding="ROUND_HALF_EVEN"
    )
    assert bin3.positive_rate == Decimal("1.000000")
