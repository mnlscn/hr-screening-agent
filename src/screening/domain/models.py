"""Domain types and the CandidateProfile model with validation and merge logic."""

from typing import Literal, Self, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from screening.domain.service_areas import load_service_area_names


DriverLicense = Literal["Yes", "No", "Pending", "Unknown"]
CityZoneStatus = Literal["Matched", "Needs clarification", "Unsupported"]
AnswerStatus = Literal["Matched", "Needs clarification"]
Availability = Literal["Full-time", "Part-time", "Weekends"]
PreferredSchedule = Literal["Morning", "Afternoon", "Evening", "Flexible"]
ConversationLanguage = Literal["English", "Spanish", "Mixed"]

BotLabel = Literal["eligible", "not_eligible", "needs_review"]
CandidateStatus = Literal["active", "completed"]

VALID_BOT_LABELS: frozenset[str] = frozenset(get_args(BotLabel))

ACTIVE_STATUS: CandidateStatus = "active"
COMPLETED_STATUS: CandidateStatus = "completed"
VALID_STATUSES: frozenset[str] = frozenset(get_args(CandidateStatus))
FINAL_STATUSES: frozenset[str] = frozenset({COMPLETED_STATUS})

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
    "full_name_status",
    "raw_drivers_license",
    "raw_city_zone",
    "city_zone_status",
    "raw_availability",
    "availability_status",
    "raw_preferred_schedule",
    "preferred_schedule_status",
    "raw_prior_delivery_experience",
    "prior_delivery_experience_status",
    "raw_start_date",
    "start_date_status",
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
            when unknown. Uses 0 for explicit "no experience" answers.
        platform (str | None): Delivery platform the candidate worked with, or
            None when unknown.
    """

    model_config = ConfigDict(extra="forbid")

    years: float | None = None
    platform: str | None = None

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
        full_name_status (AnswerStatus | None): Whether the provided name is
            complete or needs a surname follow-up.
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
        raw_availability (str | None): Verbatim availability answer when the
            candidate answered this field.
        availability_status (AnswerStatus | None): Whether availability was
            confidently normalized or needs clarification.
        preferred_schedule (PreferredSchedule | None): Normalized schedule:
            "Morning", "Afternoon", "Evening", or "Flexible".
        raw_preferred_schedule (str | None): Verbatim preferred-schedule answer.
        preferred_schedule_status (AnswerStatus | None): Whether schedule was
            confidently normalized or needs clarification.
        prior_delivery_experience (DeliveryExperience | None): Prior delivery
            experience details.
        raw_prior_delivery_experience (str | None): Verbatim delivery-experience
            answer.
        prior_delivery_experience_status (AnswerStatus | None): Whether
            delivery experience was fully captured or needs clarification.
        start_date (str | None): Candidate's reported start date.
        raw_start_date (str | None): Verbatim start-date answer.
        start_date_status (AnswerStatus | None): Whether the start date is
            actionable or needs clarification.
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
    full_name_status: AnswerStatus | None = None
    raw_drivers_license: str | None = None
    drivers_license: DriverLicense | None = None
    raw_city_zone: str | None = None
    city_zone: str | None = None
    city_zone_status: CityZoneStatus | None = None
    conversation_language: ConversationLanguage | None = None
    raw_availability: str | None = None
    availability: Availability | None = None
    availability_status: AnswerStatus | None = None
    raw_preferred_schedule: str | None = None
    preferred_schedule: PreferredSchedule | None = None
    preferred_schedule_status: AnswerStatus | None = None
    raw_prior_delivery_experience: str | None = None
    prior_delivery_experience: DeliveryExperience | None = None
    prior_delivery_experience_status: AnswerStatus | None = None
    raw_start_date: str | None = None
    start_date: str | None = None
    start_date_status: AnswerStatus | None = None
    is_complete: bool = False
    is_disqualified: bool = False
    disqualification_reasons: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    clarification_fields: list[str] = Field(default_factory=list)

    @field_validator(
        "full_name",
        "raw_drivers_license",
        "raw_city_zone",
        "raw_availability",
        "raw_preferred_schedule",
        "raw_prior_delivery_experience",
        "raw_start_date",
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

    @model_validator(mode="after")
    def refresh_status(self) -> Self:
        """Recompute derived status fields from the current field values.

        Infers ``city_zone_status`` when unset, then recomputes
        ``missing_fields``, ``clarification_fields``,
        ``disqualification_reasons``, ``is_complete``, and ``is_disqualified``.

        Returns:
            Self: The same instance with derived fields refreshed.
        """
        full_name_status = _answer_status(
            self.full_name_status,
            has_value=has_first_and_last_name(self.full_name),
            has_answer=bool(self.full_name),
        )

        city_zone_status = self.city_zone_status
        if city_zone_status is None:
            if self.city_zone:
                city_zone_status = "Matched"
            elif self.raw_city_zone:
                city_zone_status = "Needs clarification"

        availability_status = _answer_status(
            self.availability_status,
            has_value=self.availability is not None,
            has_answer=bool(self.raw_availability),
        )
        preferred_schedule_status = _answer_status(
            self.preferred_schedule_status,
            has_value=self.preferred_schedule is not None,
            has_answer=bool(self.raw_preferred_schedule),
        )
        prior_delivery_experience_status = _answer_status(
            self.prior_delivery_experience_status,
            has_value=(
                self.prior_delivery_experience is not None
                and self.prior_delivery_experience.is_complete()
            ),
            has_answer=(
                bool(self.raw_prior_delivery_experience)
                or self.prior_delivery_experience is not None
            ),
        )
        start_date_status = _answer_status(
            self.start_date_status,
            has_value=self.start_date is not None,
            has_answer=bool(self.raw_start_date),
        )

        missing_fields = []
        for field in REQUIRED_FIELDS:
            if _field_is_missing(
                self,
                field,
                city_zone_status=city_zone_status,
                availability_status=availability_status,
                preferred_schedule_status=preferred_schedule_status,
                prior_delivery_experience_status=prior_delivery_experience_status,
                start_date_status=start_date_status,
            ):
                missing_fields.append(field)

        clarification_fields = []
        if full_name_status == "Needs clarification":
            clarification_fields.append("full_name")
        if self.drivers_license in {"Pending", "Unknown"}:
            clarification_fields.append("drivers_license")
        elif self.drivers_license is None and self.raw_drivers_license:
            clarification_fields.append("drivers_license")
        if city_zone_status == "Needs clarification":
            clarification_fields.append("city_zone")
        if availability_status == "Needs clarification":
            clarification_fields.append("availability")
        if preferred_schedule_status == "Needs clarification":
            clarification_fields.append("preferred_schedule")
        if prior_delivery_experience_status == "Needs clarification":
            clarification_fields.append("prior_delivery_experience")
        if start_date_status == "Needs clarification":
            clarification_fields.append("start_date")

        reasons = []
        if self.drivers_license == "No":
            reasons.append("driver_license_no")
        if city_zone_status == "Unsupported":
            reasons.append("outside_service_area")

        object.__setattr__(self, "full_name_status", full_name_status)
        object.__setattr__(self, "city_zone_status", city_zone_status)
        object.__setattr__(self, "availability_status", availability_status)
        object.__setattr__(self, "preferred_schedule_status", preferred_schedule_status)
        object.__setattr__(
            self,
            "prior_delivery_experience_status",
            prior_delivery_experience_status,
        )
        object.__setattr__(self, "start_date_status", start_date_status)
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

        for status_field, value_field in (
            ("full_name_status", "full_name"),
            ("city_zone_status", "city_zone"),
            ("availability_status", "availability"),
            ("preferred_schedule_status", "preferred_schedule"),
            ("prior_delivery_experience_status", "prior_delivery_experience"),
            ("start_date_status", "start_date"),
        ):
            if getattr(updates, status_field) == "Needs clarification":
                current[value_field] = getattr(updates, value_field)

        if updates.city_zone_status == "Unsupported":
            current["city_zone"] = updates.city_zone

        return CandidateProfile.model_validate(current)


def _answer_status(
    status: AnswerStatus | None,
    *,
    has_value: bool,
    has_answer: bool,
) -> AnswerStatus | None:
    """Infer a reusable answer status from canonical and raw evidence."""
    if has_value:
        return status or "Matched"
    if status == "Matched":
        return "Needs clarification" if has_answer else None
    if status is None and has_answer:
        return "Needs clarification"
    return status


def _field_is_missing(
    profile: CandidateProfile,
    field: str,
    *,
    city_zone_status: CityZoneStatus | None,
    availability_status: AnswerStatus | None,
    preferred_schedule_status: AnswerStatus | None,
    prior_delivery_experience_status: AnswerStatus | None,
    start_date_status: AnswerStatus | None,
) -> bool:
    """Report whether a required field has no answer evidence at all."""
    if field == "full_name":
        return profile.full_name is None
    if field == "drivers_license":
        return profile.drivers_license is None and profile.raw_drivers_license is None
    if field == "city_zone":
        return profile.city_zone is None and city_zone_status is None
    if field == "availability":
        return profile.availability is None and availability_status is None
    if field == "preferred_schedule":
        return profile.preferred_schedule is None and preferred_schedule_status is None
    if field == "prior_delivery_experience":
        return (
            profile.prior_delivery_experience is None
            and prior_delivery_experience_status is None
        )
    if field == "start_date":
        return profile.start_date is None and start_date_status is None
    return getattr(profile, field) is None
