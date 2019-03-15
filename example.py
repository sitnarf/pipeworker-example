from pipeworker.blocks.Map import Map
from pipeworker.blocks.datasets.CompareMeasurementAndPrint import CompareMeasurementAndPrint
from pipeworker.blocks.datasets.PrintMeasurements import PrintMeasurements
from pipeworker.blocks.datasets.Measure import Measure
from pipeworker.blocks.datasets.TrainTestSplit import TrainTestSplit
from pipeworker.base.Pipeline import Pipeline
from pipeworker.functions.measurements import mae, mape
from blocks import LoadData, FillNaN, SES, SARIMA
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pipe = Pipeline(
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
)

result = pipe.execute()
