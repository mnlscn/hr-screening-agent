"""Domain types and the CandidateProfile model with validation and merge logic."""

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
    """Check whether a name string contains at least two alphabetic parts.

    Args:
        value (str | None): The candidate name to inspect.

    Returns:
        bool: True if the value has two or more whitespace-separated parts
            that each contain a letter, otherwise False.
    """
    if value is None:
        return False
    parts = [
        part for part in value.split() if any(character.isalpha() for character in part)
    ]
    return len(parts) >= 2


class DeliveryExperience(BaseModel):
    """Prior delivery-driving experience reported by a candidate.

    Attributes:
        years (float | None): Number of years of delivery experience, or None
            when unknown. Normalized to 0 for explicit "no experience" answers.
        platform (str | None): Delivery platform the candidate worked with, or
            None when unknown.
    """

    model_config = ConfigDict(extra="ignore")

    years: float | None = None
    platform: str | None = None

    @field_validator("years", mode="before")
    @classmethod
    def normalize_years(cls, value):
        """Normalize raw years input into a float or None.

        Args:
            value: Raw years value, which may be a number, an empty value, or
                a free-text "no experience" phrase in English or Spanish.

        Returns:
            The original value, None for empty input, or 0 when the text
            indicates the candidate has no experience.
        """
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
        """Strip and normalize the platform value to a non-empty string or None.

        Args:
            value: Raw platform value.

        Returns:
            The stripped platform string, or None when empty or missing.
        """
        if value is None:
            return None
        value = str(value).strip()
        return value or None

    def is_complete(self) -> bool:
        """Report whether the experience answer is fully specified.

        Returns:
            bool: True when the candidate reported no experience (years == 0)
                or when both years and platform are present.
        """
        if self.years == 0:
            return True
        return self.years is not None and bool(self.platform)


class CandidateProfile(BaseModel):
    """Structured screening profile for a delivery-driver candidate.

    Raw fields hold the candidate's verbatim answers; normalized fields hold
    canonical values. The status fields (``missing_fields``,
    ``clarification_fields``, ``disqualification_reasons``, ``is_complete``,
    ``is_disqualified``, and ``city_zone_status``) are derived automatically by
    the after-validator and should not be set manually.

    Attributes:
        full_name (str | None): Candidate's full name; considered complete only
            with both a first and last name.
        raw_drivers_license (str | None): Verbatim driver-license answer.
        drivers_license (DriverLicense | None): Normalized license status: one
            of "Yes", "No", "Pending", or "Unknown".
        raw_city_zone (str | None): Verbatim city or zone answer.
        city_zone (str | None): Canonical service-area city, validated against
            the supported service areas.
        city_zone_status (CityZoneStatus | None): Match outcome for the city:
            "Matched", "Needs clarification", or "Unsupported".
        conversation_language (ConversationLanguage | None): Predominant
            conversation language: "English", "Spanish", or "Mixed".
        availability (Availability | None): Normalized availability:
            "Full-time", "Part-time", or "Weekends".
        preferred_schedule (PreferredSchedule | None): Normalized schedule:
            "Morning", "Afternoon", "Evening", or "Flexible".
        prior_delivery_experience (DeliveryExperience | None): Prior delivery
            experience details.
        start_date (str | None): Candidate's reported start date.
        is_complete (bool): True when no fields are missing or need
            clarification. Derived.
        is_disqualified (bool): True when any disqualification reason applies.
            Derived.
        disqualification_reasons (list[str]): Hard-rule failure reasons.
            Derived.
        missing_fields (list[str]): Required fields not yet collected. Derived.
        clarification_fields (list[str]): Fields needing follow-up
            clarification. Derived.
    """

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
        """Strip optional free-text fields to a non-empty string or None.

        Args:
            value: Raw field value.

        Returns:
            The stripped string, or None when empty or missing.
        """
        if value is None:
            return None
        value = str(value).strip()
        return value or None

    @field_validator("drivers_license", mode="before")
    @classmethod
    def normalize_drivers_license(cls, value):
        """Map a free-text license answer to a canonical license status.

        Recognizes English and Spanish phrasings as well as boolean input.

        Args:
            value: Raw driver-license answer.

        Returns:
            One of "Yes", "No", "Pending", "Unknown", or the original value
            when no mapping applies, or None when missing.
        """
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
        """Ensure ``city_zone`` is a recognized canonical service area.

        Args:
            value: Normalized city value.

        Returns:
            The validated city value, or None when missing.

        Raises:
            ValueError: If the value is not one of the canonical service areas.
        """
        if value is None:
            return None
        if value not in load_service_area_names():
            raise ValueError("city_zone must be a canonical service area")
        return value

    @field_validator("availability", mode="before")
    @classmethod
    def normalize_availability(cls, value):
        """Map a free-text availability answer to a canonical value.

        Recognizes English and Spanish phrasings.

        Args:
            value: Raw availability answer.

        Returns:
            One of "Full-time", "Part-time", "Weekends", or the original value
            when no mapping applies, or None when missing.
        """
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
        """Map a free-text schedule answer to a canonical value.

        Recognizes English and Spanish phrasings.

        Args:
            value: Raw preferred-schedule answer.

        Returns:
            One of "Morning", "Afternoon", "Evening", "Flexible", or the
            original value when no mapping applies, or None when missing.
        """
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
        """Recompute derived status fields from the current field values.

        Infers ``city_zone_status`` when unset, then recomputes
        ``missing_fields``, ``clarification_fields``,
        ``disqualification_reasons``, ``is_complete``, and ``is_disqualified``.

        Returns:
            Self: The same instance with derived fields refreshed.
        """
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
        """Return a new profile with non-null updates applied over this one.

        Only fields in ``PROFILE_UPDATE_FIELDS`` are merged, and only when the
        update value is not None. When the update marks the city as needing
        clarification or unsupported, the city value is overwritten regardless.
        The merged data is re-validated, refreshing all derived status fields.

        Args:
            updates (CandidateProfile): Profile carrying the new values to
                overlay onto this profile.

        Returns:
            CandidateProfile: A new, re-validated profile reflecting the merge.
        """
        current = self.model_dump()
        for field in PROFILE_UPDATE_FIELDS:
            value = getattr(updates, field)
            if value is not None:
                current[field] = value

        if updates.city_zone_status in {"Needs clarification", "Unsupported"}:
            current["city_zone"] = updates.city_zone

        return CandidateProfile.model_validate(current)
