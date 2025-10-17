# session_util.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# adjust this if your connection uses username/password/host
DATABASE_URI = "postgresql:///stocks"

# Create engine and session factory
engine = create_engine(DATABASE_URI, echo=False)
SessionLocal = sessionmaker(bind=engine)

def get_session():
    """Return a new standalone SQLAlchemy session (no Flask app context)."""
    return SessionLocal()
