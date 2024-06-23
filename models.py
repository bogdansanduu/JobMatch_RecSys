from sqlalchemy import Column, Integer, String, Text, Float, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Company(Base):
    __tablename__ = 'company'
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True)
    password = Column(String)
    name = Column(String, unique=True)
    profile_picture = Column(Text)
    industry = Column(String)
    country = Column(String)
    state = Column(String, nullable=True)
    city = Column(String, nullable=True)

    # Relationships
    jobs = relationship("Job", back_populates="company")


class Job(Base):
    __tablename__ = 'job'
    id = Column(Integer, primary_key=True)
    title = Column(String)
    description = Column(Text)
    category = Column(String)
    country = Column(String)
    state = Column(String, nullable=True)
    city = Column(String, nullable=True)
    lat = Column(Float)
    lng = Column(Float)
    responsibilities = Column(Text)
    minimum_qualifications = Column(Text)
    preferred_qualifications = Column(Text)
    company_id = Column(Integer, ForeignKey('company.id'))

    # Relationship
    company = relationship("Company", back_populates="jobs")