import os
from fastapi import HTTPException, Request
from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, select

from models import Job, Company

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
SECRET_EXPRESS_SERVER = os.getenv("SECRET_EXPRESS_SERVER")

engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine)


def get_data():
    session = SessionLocal()
    try:
        result = session.execute(
            select(Job).join(Company, Job.company_id == Company.id)
        )
        jobs_data = result.scalars().all()

        jobs_list = []
        for job in jobs_data:
            jobs_list.append({
                'id': job.id,
                'company': job.company.name,
                'title': job.title,
                'category': job.category,
                'responsibilities': job.responsibilities,
                'minimum_qualifications': job.minimum_qualifications,
                'preferred_qualifications': job.preferred_qualifications,
                'latitude': job.lat,
                'longitude': job.lng
            })
    finally:
        session.close()

    return jobs_list


def verify_secret_key(request: Request):
    secret_header = request.headers.get('X-Secret-Key')

    if secret_header != SECRET_EXPRESS_SERVER:
        raise HTTPException(status_code=403, detail="Unauthorized")