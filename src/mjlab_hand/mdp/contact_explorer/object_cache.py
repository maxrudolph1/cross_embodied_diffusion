"""ContactDB object cache for ContactExplorer managers."""

from __future__ import annotations

from mjlab_hand.object.object import ContactDBObject

_OBJECT_CACHE: dict[str, ContactDBObject] = {}


def get_cached_contact_db_object(object_name: str, **kwargs) -> ContactDBObject:
    scale = kwargs.get("scale", (1.0, 1.0, 1.0))
    key = (
        f"{object_name}|{kwargs.get('num_points', 2048)}|{kwargs.get('num_regions', 64)}|{scale!r}"
    )
    if key not in _OBJECT_CACHE:
        _OBJECT_CACHE[key] = ContactDBObject(object_name, **kwargs)
    return _OBJECT_CACHE[key]
