"""Release owned resources without replacing an earlier operation failure."""
from collections.abc import Callable

from app.exceptions import ResourceCleanupError


def close_resource(close: Callable[[], None], failure: BaseException | None = None) -> None:
    try:
        close()
    except Exception:
        if failure is not None:
            failure.add_note('Resource cleanup failed; resources may remain open.')
        else:
            raise ResourceCleanupError('Resource cleanup failed; resources may remain open.') from None
