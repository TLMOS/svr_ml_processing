from redis import Redis

from common.config import settings


connection_args = {
    'host': str(settings.redis.host),
    'port': settings.redis.port,
    'db': settings.redis.db,
}

if settings.redis.username:
    connection_args['username'] = settings.redis.username
if settings.redis.password:
    connection_args['password'] = settings.redis.password

redis = Redis(**connection_args)
