import pytest
import pytest_asyncio
import pandas as pd

from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
from starlette.testclient import TestClient

from backend.core.database import Base, get_db
from backend.main import app


# StaticPool forces aiosqlite to reuse a single in-memory connection across
# all operations in the test — without it each new connection gets a blank DB
_IN_MEMORY_URL = "sqlite+aiosqlite:///:memory:"


def _make_test_engine():
    return create_async_engine(
        _IN_MEMORY_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


@pytest.fixture
def small_csv_path(tmp_path):
    # Write a deterministic 20-row CSV to tmp_path and return its Path
    rows = [
        {"age": 20 + i, "city": "hanoi" if i % 2 == 0 else "hcm", "score": round(70.0 + i * 1.5, 1)}
        for i in range(20)
    ]
    path = tmp_path / "small_test.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest_asyncio.fixture
async def async_db():
    # Yield an AsyncSession backed by a fresh in-memory SQLite DB with all tables created
    engine = _make_test_engine()

    # Populate metadata by importing models before create_all
    from backend.core import models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with session_factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest.fixture
def test_client(async_db):
    # Return a TestClient with the get_db dependency overridden to use the in-memory session
    async def _override_get_db():
        yield async_db

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client
    app.dependency_overrides.clear()