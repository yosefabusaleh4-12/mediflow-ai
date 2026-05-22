from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.orm import declarative_base, sessionmaker

# Create database file
engine = create_engine("sqlite:///mediflow.db")

Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

# 🧠 This is your memory table
class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)

    quantity = Column(Integer)
    usage = Column(Integer)
    days_left = Column(Integer)
    cost = Column(Integer)

    expiry_risk = Column(Float)
    shortage_risk = Column(Float)

    expiry_action = Column(String)
    shortage_action = Column(String)

# Create table automatically
Base.metadata.create_all(bind=engine)

print("Database ready")