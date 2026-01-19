import time
from config.settings import MAX_VIOLATIONS

class ViolationManager:
    def __init__(self):
        self.attempts = 0
        self.records = []
        self.active_violations = []  # List of (violation_text, timestamp) tuples
        self.violation_display_duration = 3  # seconds

    def register(self, violation):
        self.attempts += 1
        self.records.append(violation)
        # Add violation with current timestamp for display
        self.active_violations.append((violation, time.time()))
        print(f"[VIOLATION {self.attempts}] {violation}")

        return self.attempts <= MAX_VIOLATIONS

    def get_active_violations(self):
        """Get list of violations that are still active (within 3 seconds)"""
        current_time = time.time()
        # Filter out expired violations
        self.active_violations = [
            (v, ts) for v, ts in self.active_violations
            if current_time - ts < self.violation_display_duration
        ]
        # Return just the violation text in reverse order (newest at bottom)
        return [v for v, ts in self.active_violations]
