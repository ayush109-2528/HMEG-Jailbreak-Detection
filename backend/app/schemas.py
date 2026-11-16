from pydantic import BaseModel, Field, field_validator, model_validator


class PredictionInput(BaseModel):
    prompt: str = Field(min_length=1, max_length=10000)
    public_consent: bool = False
    low_threshold: float | None = Field(default=None, ge=0, le=1)
    high_threshold: float | None = Field(default=None, ge=0, le=1)
    final_threshold: float | None = Field(default=None, gt=0, lt=1)

    @model_validator(mode='after')
    def experimental_thresholds(self):
        values = (self.low_threshold, self.high_threshold, self.final_threshold)
        if any(v is not None for v in values):
            if not all(v is not None for v in values):
                raise ValueError('Provide all three experimental thresholds together.')
            if not self.low_threshold < self.final_threshold < self.high_threshold:
                raise ValueError('Require low < final < high.')
        return self


    @field_validator('prompt')
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError('Enter a nonempty prompt.')
        return value
