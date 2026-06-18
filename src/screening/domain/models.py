from typing import Literal, Self, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from screening.domain.service_areas import load_service_area_names


DriverLicense = Literal["Yes", "No", "Pending", "Unknown"]
CityZoneStatus = Literal["Matched", "Needs clarification", "Unsupported"]
Availability = Literal["Full-time", "Part-time", "Weekends"]
PreferredSchedule = Literal["Morning", "Afternoon", "Evening", "Flexible"]
ConversationLanguage = Literal["English", "Spanish", "Mixed"]

BotLabel = Literal["eligible", "not_eligible", "needs_review"]
CandidateStatus = Literal["active", "completed", "disqualified"]

VALID_BOT_LABELS: frozenset[str] = frozenset(get_args(BotLabel))

ACTIVE_STATUS: CandidateStatus = "active"
COMPLETED_STATUS: CandidateStatus = "completed"
DISQUALIFIED_STATUS: CandidateStatus = "disqualified"
VALID_STATUSES: frozenset[str] = frozenset(get_args(CandidateStatus))
FINAL_STATUSES: frozenset[str] = frozenset({COMPLETED_STATUS, DISQUALIFIED_STATUS})

REQUIRED_FIELDS = (
    "full_name",
    "drivers_license",
    "city_zone",
    "availability",
    "preferred_schedule",
    "prior_delivery_experience",
    "start_date",
)

PROFILE_UPDATE_FIELDS = REQUIRED_FIELDS + (
    "raw_drivers_license",
    "raw_city_zone",
    "city_zone_status",
    "conversation_language",
)


def has_first_and_last_name(value: str | None) -> bool:
    if value is None:
        return False
    parts = [
        part for part in value.split() if any(character.isalpha() for character in part)
    ]
    return len(parts) >= 2


class DeliveryExperience(BaseModel):
    model_config = ConfigDict(extra="ignore")

    years: float | None = None
    platform: str | None = None

    @field_validator("years", mode="before")
    @classmethod
    def normalize_years(cls, value):
        if value in (None, ""):
            return None
        if isinstance(value, str) and value.strip().lower() in {
            "no",
            "nope",
            "nah",
            "none",
            "ninguna",
            "ninguno",
            "sin experiencia",
            "no experience",
        }:
            return 0
        return value

    @field_validator("platform", mode="before")
    @classmethod
    def normalize_platform(cls, value):
        if value is None:
            return None
        value = str(value).strip()
        return value or None

    def is_complete(self) -> bool:
        if self.years == 0:
            return True
        return self.years is not None and bool(self.platform)


class CandidateProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    full_name: str | None = None
    raw_drivers_license: str | None = None
    drivers_license: DriverLicense | None = None
    raw_city_zone: str | None = None
    city_zone: str | None = None
    city_zone_status: CityZoneStatus | None = None
    conversation_language: ConversationLanguage | None = None
    availability: Availability | None = None
    preferred_schedule: PreferredSchedule | None = None
    prior_delivery_experience: DeliveryExperience | None = None
    start_date: str | None = None
    is_complete: bool = False
    is_disqualified: bool = False
    disqualification_reasons: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    clarification_fields: list[str] = Field(default_factory=list)

    @field_validator(
        "full_name",
        "raw_drivers_license",
        "raw_city_zone",
        "city_zone",
        "start_date",
        mode="before",
    )
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
        if normalized in {"no", "n", "nope", "nah", "false"}:
            return "No"
        if normalized in {"pending", "in progress", "taking it soon", "tramitando"}:
            return "Pending"
        if normalized in {"unknown", "unclear", "not clear"}:
            return "Unknown"
        if any(
            phrase in normalized
            for phrase in (
                "taking it",
                "next week",
                "soon",
                "in process",
                "in progress",
                "sacando",
                "en tramite",
                "en trámite",
                "la semana que viene",
            )
        ):
            return "Pending"
        if any(
            phrase in normalized
            for phrase in (
                "i have one",
                "i have it",
                "got one",
                "tengo carnet",
                "tengo licencia",
                "tengo permiso",
            )
        ):
            return "Yes"
        return value

    @field_validator("city_zone")
    @classmethod
    def validate_canonical_city_zone(cls, value):
        if value is None:
            return None
        if value not in load_service_area_names():
            raise ValueError("city_zone must be a canonical service area")
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
        city_zone_status = self.city_zone_status
        if city_zone_status is None:
            if self.city_zone:
                city_zone_status = "Matched"
            elif self.raw_city_zone:
                city_zone_status = "Needs clarification"

        missing_fields = []
        for field in REQUIRED_FIELDS:
            value = getattr(self, field)
            if value is None:
                missing_fields.append(field)
            elif field == "full_name" and not has_first_and_last_name(value):
                missing_fields.append(field)
            elif field == "prior_delivery_experience" and not value.is_complete():
                missing_fields.append(field)

        clarification_fields = []
        if self.drivers_license in {"Pending", "Unknown"}:
            clarification_fields.append("drivers_license")
        if city_zone_status == "Needs clarification":
            clarification_fields.append("city_zone")

        reasons = []
        if self.drivers_license == "No":
            reasons.append("driver_license_no")
        if city_zone_status == "Unsupported":
            reasons.append("outside_service_area")

        object.__setattr__(self, "city_zone_status", city_zone_status)
        object.__setattr__(self, "missing_fields", missing_fields)
        object.__setattr__(self, "clarification_fields", clarification_fields)
        object.__setattr__(self, "disqualification_reasons", reasons)
        object.__setattr__(
            self,
            "is_complete",
            not missing_fields and not clarification_fields,
        )
        object.__setattr__(self, "is_disqualified", bool(reasons))
        return self

    def merge(self, updates: "CandidateProfile") -> "CandidateProfile":
        current = self.model_dump()
        for field in PROFILE_UPDATE_FIELDS:
            value = getattr(updates, field)
            if value is not None:
                current[field] = value

        if updates.city_zone_status in {"Needs clarification", "Unsupported"}:
            current["city_zone"] = updates.city_zone

        return CandidateProfile.model_validate(current)
