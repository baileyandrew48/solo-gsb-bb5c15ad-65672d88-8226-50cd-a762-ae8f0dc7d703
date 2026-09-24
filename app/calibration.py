"""跨事件概率校准报告：把已结算 0/1 事件的揭示原文当作概率预测做精确校准分析。

纯函数模块：不依赖数据库与 HTTP，可独立单测。

概率原文语法（严格匹配，不允许空白、符号、指数等宽松形式）：

    probability := 数字+ ["." 数字*] | "." 数字+

即只接受**无符号、无指数、无空白**的十进制记法；解析出的数值还必须位于
闭区间 [0, 1]（0 与 1 边界均有效）。无效原文分两类注明原因：
``INVALID_FORMAT``（不是上述纯十进制记法）与 ``OUT_OF_RANGE``（数值不在
[0, 1]）。

每条有效预测的平方误差 ``(概率 - 实际结果)^2``、总体 Brier 均分（全部有效
预测平方误差的算术平均）以及各概率区间的预测均值与「实际为 1」频率，均用
``decimal.Decimal`` 精确计算，全程不经过浮点；对外统一四舍六入五成双
（ROUND_HALF_EVEN）到 6 位小数。
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from uuid import UUID

# 无空白 / 无符号 / 无指数的纯十进制记法：
#   至少含一位数字；可选小数点，但 "." 单独不合法
# 显式排除 Decimal 构造器宽容接受的空白、符号、下划线、指数与 Infinity/NaN
PROBABILITY_PATTERN_TEXT = r"^(?:\d+(?:\.\d*)?|\.\d+)$"

# 无效原因
INVALID_FORMAT = "INVALID_FORMAT"  # 不是无空白 / 无符号 / 无指数的纯十进制记法
OUT_OF_RANGE = "OUT_OF_RANGE"  # 数值不在闭区间 [0, 1]

ZERO = Decimal("0")
ONE = Decimal("1")

# 对外数值统一输出的 6 位小数定点标度
SIX_PLACES = Decimal("0.000001")

# 十个校准区间的下界：[0,0.1)、[0.1,0.2)、…、[0.8,0.9)、[0.9,1]
BIN_EDGES = [Decimal(i) / Decimal(10) for i in range(10)]


def quantize_six(value: Decimal) -> Decimal:
    """四舍六入五成双（银行家舍入）到 6 位小数的定点 Decimal。"""
    return value.quantize(SIX_PLACES, rounding=ROUND_HALF_EVEN)


def parse_probability(text: str) -> tuple[Decimal | None, str | None]:
    """把揭示原文解析为 [0, 1] 内的概率。

    返回 (概率, None) 或 (None, 原因)；原因取 INVALID_FORMAT（不是无空白 /
    无符号 / 无指数的纯十进制记法）或 OUT_OF_RANGE（数值不在 [0, 1]）。
    """
    # 以正则保证语法，再交给 Decimal：纯数字记法不会构造出非有限值，
    # 但超长数字串可能超出 Decimal 可表示范围，统一按格式非法处理
    if re.fullmatch(PROBABILITY_PATTERN_TEXT, text) is None:
        return None, INVALID_FORMAT
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None, INVALID_FORMAT
    if not value.is_finite() or value < ZERO or value > ONE:
        return None, OUT_OF_RANGE
    return value, None


@dataclass(frozen=True)
class CalibrationPrediction:
    """一条有效概率预测：概率为精确十进制，平方误差已量化到 6 位小数。"""

    event_id: UUID
    commitment_id: UUID
    committed_at: datetime
    original_text: str
    probability: Decimal  # 原文解析出的精确概率（[0, 1]）
    actual: int  # 结算标准结果 0 | 1
    squared_error: Decimal  # (概率 - 实际)^2，已量化到 6 位小数


@dataclass(frozen=True)
class InvalidPrediction:
    """一条无法作为 [0, 1] 概率使用的揭示原文。"""

    event_id: UUID
    commitment_id: UUID
    committed_at: datetime
    original_text: str
    reason: str  # INVALID_FORMAT | OUT_OF_RANGE


@dataclass(frozen=True)
class CalibrationBin:
    """一个概率区间的汇总（边界归属确定，空样本均值/频率为 None）。"""

    index: int  # 0～9
    lower_inclusive: Decimal
    upper_exclusive: Decimal  # 最后一个区间（index=9）上界闭于 1
    mean_prediction: Decimal | None
    positive_rate: Decimal | None
    sample_count: int


@dataclass(frozen=True)
class CalibrationReport:
    """校准报告的确定性结果：条目与区间均为固定顺序。"""

    valid_predictions: list[CalibrationPrediction]  # 按 (事件序, 承诺提交时间, 承诺 ID)
    invalid_predictions: list[InvalidPrediction]  # 同上
    bins: list[CalibrationBin] = field(default_factory=list)
    unrevealed_count: int = 0  # 这些事件中承诺存在但从未揭示的总数
    valid_count: int = 0
    invalid_count: int = 0
    brier_mean: Decimal | None = None  # 有效预测平方误差的算术平均；无有效样本为 None


def bin_index_for(probability: Decimal) -> int:
    """按固定边界归属返回区间下标：[0,0.1)→0 … [0.8,0.9)→8，[0.9,1]→9。"""
    index = int(probability * 10)
    if index > 9:  # 上界 1（含）归入最后一个区间
        index = 9
    return index


def build_calibration_report(
    outcomes: list[tuple[UUID, int]],
    revealed: list[tuple[UUID, UUID, datetime, str]],
    unrevealed_count: int,
) -> CalibrationReport:
    """跨事件汇总有效/无效概率预测并计算 Brier 均分与十区间校准。

    - outcomes：去重后按请求首次出现顺序排列的 (事件 ID, 实际结果 0/1)；
    - revealed：这些事件的 (事件 ID, 承诺 ID, 承诺提交时间, 揭示原文)，
      输入顺序任意；
    - 有效预测按 (事件在 outcomes 中的顺序, 承诺提交时间, 承诺 ID) 固定
      排序，无效预测使用相同排序；
    - 平方误差按条量化到 6 位小数；Brier 均分在精确误差上取算术平均后再
      量化（无有效样本时为 None）；
    - 区间归属：前九个为左闭右开，最后一个 [0.9, 1] 含上界；空样本区间的
      预测均值与「实际为 1」频率均为 None。
    """
    outcome_index = {event_id: i for i, (event_id, _) in enumerate(outcomes)}
    outcomes_by_id = dict(outcomes)

    valid: list[CalibrationPrediction] = []
    invalid: list[InvalidPrediction] = []
    for event_id, commitment_id, committed_at, original_text in revealed:
        probability, reason = parse_probability(original_text)
        if probability is None:
            invalid.append(
                InvalidPrediction(
                    event_id=event_id,
                    commitment_id=commitment_id,
                    committed_at=committed_at,
                    original_text=original_text,
                    reason=reason,
                )
            )
            continue
        actual = outcomes_by_id[event_id]
        delta = probability - Decimal(actual)
        valid.append(
            CalibrationPrediction(
                event_id=event_id,
                commitment_id=commitment_id,
                committed_at=committed_at,
                original_text=original_text,
                probability=probability,
                actual=actual,
                squared_error=quantize_six(delta * delta),
            )
        )

    valid.sort(key=lambda p: (outcome_index[p.event_id], p.committed_at, p.commitment_id.int))
    invalid.sort(
        key=lambda p: (outcome_index[p.event_id], p.committed_at, p.commitment_id.int)
    )

    # 平方误差之和需要未量化的精确误差：用原始概率重算，避免先舍入再平均
    # 带来双重舍入；Brier 均分 = 精确平方误差的算术平均，最后统一量化
    if valid:
        exact_sum = sum(
            (p.probability - Decimal(p.actual)) ** 2 for p in valid
        )
        brier_mean: Decimal | None = quantize_six(exact_sum / Decimal(len(valid)))
    else:
        brier_mean = None

    # 十区间汇总：先按下标归集，再按固定顺序输出
    sums: list[Decimal] = [ZERO] * 10
    positives = [0] * 10
    counts = [0] * 10
    for p in valid:
        i = bin_index_for(p.probability)
        sums[i] += p.probability
        positives[i] += p.actual
        counts[i] += 1

    bins: list[CalibrationBin] = []
    for i, edge in enumerate(BIN_EDGES):
        if counts[i]:
            mean = quantize_six(sums[i] / Decimal(counts[i]))
            rate = quantize_six(Decimal(positives[i]) / Decimal(counts[i]))
        else:
            mean = None
            rate = None
        bins.append(
            CalibrationBin(
                index=i,
                lower_inclusive=edge,
                upper_exclusive=edge + Decimal("0.1"),
                mean_prediction=mean,
                positive_rate=rate,
                sample_count=counts[i],
            )
        )

    return CalibrationReport(
        valid_predictions=valid,
        invalid_predictions=invalid,
        bins=bins,
        unrevealed_count=unrevealed_count,
        valid_count=len(valid),
        invalid_count=len(invalid),
        brier_mean=brier_mean,
    )
