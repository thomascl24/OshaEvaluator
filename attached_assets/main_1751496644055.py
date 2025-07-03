from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from nli_subapp import subapp, lifespan_mechanism

@asynccontextmanager
async def main_lifespan(app: FastAPI):
    async with AsyncExitStack() as stack:
        # Manage the lifecycle of sub_app
        await stack.enter_async_context(
            lifespan_mechanism(subapp)
        )
        yield

app = FastAPI(lifespan=main_lifespan)

app.mount("/app", subapp)
