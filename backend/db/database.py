from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from .models import Base

DATABASE_URL = "sqlite+aiosqlite:///./calibrai.db"

engine = create_async_engine(DATABASE_URL, echo=False)
async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db() -> AsyncSession:
    async with async_session_factory() as session:
        yield session


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
