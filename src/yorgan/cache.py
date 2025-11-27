from __future__ import annotations

from typing import TYPE_CHECKING, Awaitable, Callable, Concatenate, Optional, ParamSpec, TypeVar
import asyncio
import functools
import inspect
import os
from pathlib import Path

from aiocache import BaseCache, SimpleMemoryCache
from aiocache.base import _ensure_key

if TYPE_CHECKING:
    from yorgan.services.base import BaseService, T


@functools.lru_cache(maxsize=1)
def is_dev_env():
    # dev or prod
    APP_ENV = os.getenv("APP_ENV", "dev")
    if APP_ENV == "dev":
        return True
    else:
        return False


class SimpleMemoryCacheWithPersistence(SimpleMemoryCache):

    def __init__(self, persist_dir: Optional[str | Path] = None, serializer=None, **kwargs):
        if not persist_dir:
            persist_dir = os.getenv("LOCAL_CACHE_DIR", "./")

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(serializer=serializer, **kwargs)

    async def _set(self, key, value, ttl=None, _cas_token=None, _conn=None):
        # it is decided that ":" is namespace separator see redis convention
        json_filename_key = key + '.json'
        save_path = self.persist_dir / Path(*json_filename_key.split(':'))
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # value usually is pydantic model
        json_str = value.model_dump_json(indent=4)

        # Write file in a separate thread
        await asyncio.to_thread(
            save_path.write_text,
            json_str,
            encoding="utf-8"
        )
        # we save the json to disk but the pydantic model is in the cache
        return await super()._set(key=key, value=value, ttl=ttl, _cas_token=_cas_token, _conn=_conn)

    async def _clear(self, namespace=None, _conn=None):
        if namespace:
            for key in list(self._cache):
                if key.startswith(namespace):
                    self.__delete(key)
        return await super()._clear(namespace, _conn)

    def __delete(self, key):
        # delete the corresponding file
        json_filename_key = key + '.json'
        save_path = self.persist_dir / Path(*json_filename_key.split(':'))

        result = 1
        if save_path.exists():
            save_path.unlink()
            result = 0

        return result

    def _build_key(self, key, namespace=None):
        ns = namespace or self.namespace
        if ns is not None:
            return "{}:{}".format(namespace, _ensure_key(key))
        return key

    async def preload(self, service: BaseService[T], namespace: Optional[str] = None, filename_key: Optional[str] = None):

        if namespace == "auto" and service:
            ns = getattr(service, "service_name", service.__class__.__qualname__)
        else:
            ns = namespace

        service_dir_cache = self.persist_dir / Path(ns or "")
        for saved_json_path in service_dir_cache.rglob("*.json"):
            if filename_key and filename_key != saved_json_path.stem:
                continue
            ns_key = self.build_key(key=saved_json_path.stem, namespace=ns)
            loaded_object = service.response_type.model_validate_json(
                saved_json_path.read_text(encoding="utf-8")
            )
            await super()._set(key=ns_key, value=loaded_object)


class NullCache(BaseCache):
    """
   This cache does nothing. It is the default cache that gets instantiated if no cache is injected into
   particular service class

    Example:
        >>> cache = NullCache()
        >>> result = await cache.get("some_key")
        >>> print(result)
        None
        >>> set_result = await cache.set("another_key", "some_value")
        >>> print(set_result)
        False
   """

    async def get(self, *args, **kwargs):
        """Always return None"""

        return None

    async def set(self, *args, **kwargs):
        """Always return false"""
        return False


P = ParamSpec("P")
R = TypeVar("R")
Slf = TypeVar("Slf", bound="BaseService")


def cache_result(key_params: list[str]) -> Callable:
    """
    A decorator to cache the results of a function that returns a Pydantic BaseModel.

    key_params: list[str] - names of the params of the decorated function to be used for cache key
    """
    if not isinstance(key_params, list):
        raise ValueError("key_params must be a list")

    def decorator(func: Callable[Concatenate[Slf, P], Awaitable[R]]) -> Callable[Concatenate[Slf, P], Awaitable[R]]:
        if not inspect.iscoroutinefunction(func):
            raise ValueError("We expect async function")

        sig = inspect.signature(func)

        @functools.wraps(func)
        async def wrapper(self: Slf, *args: P.args, **kwargs: P.kwargs) -> R:
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()

            # Build key from only the specified parameters
            key_values = [str(bound.arguments[param_name]) for param_name in key_params]
            cache_key = "_".join(key_values)

            ns = getattr(self, "service_name", self.__class__.__qualname__)

            value = await self.cache.get(cache_key, namespace=ns)
            if value:
                if is_dev_env():
                    print(f"------ cache hit with key {cache_key}")
            else:
                value = await func(self, *args, **kwargs)
                if is_dev_env():
                    print(f"------ storing value for key {cache_key}")
                await self.cache.set(key=cache_key, value=value, namespace=ns)

            return value

        return wrapper

    return decorator
