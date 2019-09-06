import warnings

from pipeworker.base import LogLevel
from pipeworker.cache_engine import FileCacheEngine
from pipeworker.functions.measurements import mae, mape
from pipeworker.nodes.datasets.CompareMeasurementAndPrint import CompareMeasurementAndPrint
from pipeworker.nodes.datasets.Measure import Measure
from pipeworker.nodes.datasets.PrintMeasurements import PrintMeasurements
from pipeworker.nodes.datasets.TrainTestSplit import TrainTestSplit
from pipeworker.nodes.transformers import Map

from nodes import LoadData, SARIMA, SES, FillNaN


warnings.simplefilter(action='ignore', category=FutureWarning)

graph = (
        LoadData() |
        FillNaN() |
        TrainTestSplit(shuffle=False) |
        (
                SES().set_name("SES") &
                (SARIMA((2, 1, 1), (0, 1, 0, 12)).set_name("SARIMA 1")) &
                (SARIMA((2, 1, 1), (0, 0, 0, 12)).set_name("SARIMA 2"))
        ) |
        Map(Measure(measurements=[mae, mape], column="passengers")) |
        Map(PrintMeasurements()) |
        CompareMeasurementAndPrint(which="mape")
) \
    .set_log_level(LogLevel.ENABLED) \
    .set_cache_engine(FileCacheEngine("temp"))

result = graph.invoke()
