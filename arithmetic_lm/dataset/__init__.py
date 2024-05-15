from .arithmetic_dataset import (
    ArithmeticExampleDataset,
    ArithmeticLMDataset,
    ArithmeticLMSequenceDataset,
    LightningArithmeticDataModule,
)

DATASET_CLASSES = {
    "ArithmeticExampleDataset": ArithmeticExampleDataset,
    "ArithmeticLMDataset": ArithmeticLMDataset,
    "ArithmeticLMSequenceDataset": ArithmeticLMSequenceDataset,
}
