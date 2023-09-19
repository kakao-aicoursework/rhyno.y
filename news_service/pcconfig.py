import pynecone as pc

class NewsserviceConfig(pc.Config):
    pass

config = NewsserviceConfig(
    app_name="news_service",
    db_url="sqlite:///pynecone.db",
    env=pc.Env.DEV,
)