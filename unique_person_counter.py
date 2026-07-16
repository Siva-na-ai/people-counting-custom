class UniquePersonCounter:
    """Count unique persons from confirmed identity IDs only."""

    def __init__(self):
        self._seen = set()

    def update(self, person_id, state):
        if person_id is None:
            return len(self._seen)
        if state in {"CONFIRMED", "TRACK_LOCKED", "REIDENTIFIED"}:
            self._seen.add(person_id)
        return len(self._seen)

    def count(self):
        return len(self._seen)
