from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from screening.config import SERVICE_AREAS


DriverLicense = Literal["Yes", "No"]
Availability = Literal["Full-time", "Part-time", "Weekends"]
PreferredSchedule = Literal["Morning", "Afternoon", "Evening", "Flexible"]

REQUIRED_FIELDS = (
    "full_name",
    "drivers_license",
    "city_zone",
    "availability",
    "preferred_schedule",
    "prior_delivery_experience",
    "start_date",
)


class DeliveryExperience(BaseModel):
    model_config = ConfigDict(extra="ignore")

    years: float | None = None
    platform: str | None = None

    @field_validator("years", mode="before")
    @classmethod
    def normalize_years(cls, value):
        if value in (None, ""):
            return None
        return value

    @field_validator("platform", mode="before")
    @classmethod
    def normalize_platform(cls, value):
        if value is None:
            return None
        value = str(value).strip()
        return value or None

    def is_complete(self) -> bool:
        return self.years is not None and bool(self.platform)


class CandidateProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    full_name: str | None = None
    drivers_license: DriverLicense | None = None
    city_zone: str | None = None
    availability: Availability | None = None
    preferred_schedule: PreferredSchedule | None = None
    prior_delivery_experience: DeliveryExperience | None = None
    start_date: str | None = None
    is_complete: bool = False
    is_disqualified: bool = False
    disqualification_reasons: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)

    @field_validator("full_name", "city_zone", "start_date", mode="before")
    @classmethod
    def normalize_optional_text(cls, value):
        if value is None:
            return None
        value = str(value).strip()
        return value or None

    @field_validator("drivers_license", mode="before")
    @classmethod
    def normalize_drivers_license(cls, value):
        if value is None:
            return None
        if isinstance(value, bool):
            return "Yes" if value else "No"

        normalized = str(value).strip().lower()
        if normalized in {"yes", "y", "true", "si", "sí"}:
            return "Yes"
        if normalized in {"no", "n", "false"}:
            return "No"
        return value

    @field_validator("availability", mode="before")
    @classmethod
    def normalize_availability(cls, value):
        if value is None:
            return None

        normalized = str(value).strip().lower().replace("_", " ").replace("-", " ")
        values = {
            "full time": "Full-time",
            "fulltime": "Full-time",
            "tiempo completo": "Full-time",
            "part time": "Part-time",
            "parttime": "Part-time",
            "medio tiempo": "Part-time",
            "weekends": "Weekends",
            "weekend": "Weekends",
            "fines de semana": "Weekends",
            "fin de semana": "Weekends",
        }
        return values.get(normalized, value)

    @field_validator("preferred_schedule", mode="before")
    @classmethod
    def normalize_preferred_schedule(cls, value):
        if value is None:
            return None

        normalized = str(value).strip().lower()
        values = {
            "morning": "Morning",
            "mañana": "Morning",
            "manana": "Morning",
            "afternoon": "Afternoon",
            "tarde": "Afternoon",
            "evening": "Evening",
            "noche": "Evening",
            "flexible": "Flexible",
            "flex": "Flexible",
        }
        return values.get(normalized, value)

    @model_validator(mode="after")
    def refresh_status(self) -> Self:
        missing_fields = []
        for field in REQUIRED_FIELDS:
            value = getattr(self, field)
            if value is None:
                missing_fields.append(field)
            elif field == "prior_delivery_experience" and not value.is_complete():
                missing_fields.append(field)

        reasons = []
        if self.drivers_license == "No":
            reasons.append("driver_license_no")
        if self.city_zone and not self.is_service_area(self.city_zone):
            reasons.append("outside_service_area")

        object.__setattr__(self, "missing_fields", missing_fields)
        object.__setattr__(self, "disqualification_reasons", reasons)
        object.__setattr__(self, "is_complete", not missing_fields)
        object.__setattr__(self, "is_disqualified", bool(reasons))
        return self

    @staticmethod
    def is_service_area(city_zone: str) -> bool:
        normalized = city_zone.strip().lower()
        return normalized in {area.lower() for area in SERVICE_AREAS}

    def merge(self, updates: "CandidateProfile") -> "CandidateProfile":
        current = self.model_dump()
        for field in REQUIRED_FIELDS:
            value = getattr(updates, field)
            if value is not None:
                current[field] = value
        return CandidateProfile.model_validate(current)
