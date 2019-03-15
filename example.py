from tube.blocks.Map import Map
from tube.blocks.datasets.CompareMeasurementAndPrint import CompareMeasurementAndPrint
from tube.blocks.datasets.PlotResults import PlotForecast
from tube.blocks.datasets.PrintMeasurements import PrintMeasurements
from tube.blocks.datasets.Measure import Measure
from tube.blocks.datasets.TrainTestSplit import TrainTestSplit
from tube.base.Pipeline import Pipeline
from tube.functions.measurements import mae, mape
from blocks import LoadData, FillNaN, SES, SARIMA

pipe = Pipeline(
    LoadData() |
    FillNaN() |
    TrainTestSplit(shuffle=False) |
    (
            SES().set_name("SES") &
            (SARIMA((2, 1, 1), (0, 1, 0, 12)).set_name("SARIMA 1") | PlotForecast(["passengers"])) &
            (SARIMA((2, 1, 1), (0, 0, 0, 12)).set_name("SARIMA 2"))
    ) |
    Map(Measure(measurements=[mae, mape], column="passengers")) |
    Map(PrintMeasurements()) |
    CompareMeasurementAndPrint(which="mape")
)

result = pipe.execute()
