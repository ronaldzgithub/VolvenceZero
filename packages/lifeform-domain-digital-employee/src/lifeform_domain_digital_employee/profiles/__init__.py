"""Built-in industry profiles for the digital-employee vertical.

Each profile is a reviewed, frozen data bundle (see
:class:`lifeform_domain_digital_employee.IndustryProfile`). The registry
below is the discovery surface: callers resolve a profile by its exact
``industry_id`` (protocol identifier, not text matching) and feed it to
:func:`lifeform_domain_digital_employee.build_industry_package`.

Adding an industry = adding one data module here and listing its builder
in ``BUILTIN_INDUSTRY_PROFILE_BUILDERS``. No code branches anywhere else.
"""

from __future__ import annotations

from collections.abc import Callable

from lifeform_domain_digital_employee.industry import IndustryProfile
from lifeform_domain_digital_employee.profiles.content_editor import (
    build_content_editor_profile,
)
from lifeform_domain_digital_employee.profiles.customer_support import (
    build_customer_support_profile,
)
from lifeform_domain_digital_employee.profiles.sales_sdr import (
    build_sales_sdr_profile,
)

BUILTIN_INDUSTRY_PROFILE_BUILDERS: tuple[Callable[[], IndustryProfile], ...] = (
    build_sales_sdr_profile,
    build_customer_support_profile,
    build_content_editor_profile,
)


def builtin_industry_profiles() -> tuple[IndustryProfile, ...]:
    """Materialise every built-in industry profile."""

    return tuple(builder() for builder in BUILTIN_INDUSTRY_PROFILE_BUILDERS)


def industry_profile_by_id(industry_id: str) -> IndustryProfile:
    """Resolve a built-in profile by exact ``industry_id``; fail loudly."""

    profiles = builtin_industry_profiles()
    by_id = {profile.industry_id: profile for profile in profiles}
    profile = by_id.get(industry_id)
    if profile is None:
        raise KeyError(
            f"unknown digital-employee industry_id {industry_id!r}; "
            f"built-in profiles: {sorted(by_id.keys())!r}"
        )
    return profile


__all__ = [
    "BUILTIN_INDUSTRY_PROFILE_BUILDERS",
    "build_content_editor_profile",
    "build_customer_support_profile",
    "build_sales_sdr_profile",
    "builtin_industry_profiles",
    "industry_profile_by_id",
]
