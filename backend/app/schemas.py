from pydantic import BaseModel, Field, field_validator


class PredictionInput(BaseModel):
    prompt: str = Field(min_length=1, max_length=10000)
    public_consent: bool = False

    @field_validator('prompt')
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError('Enter a nonempty prompt.')
        return value
