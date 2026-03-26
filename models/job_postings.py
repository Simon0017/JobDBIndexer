from sqlalchemy.orm import relationship
from DbIndexing.models.setup import Base
from DbIndexing.models.setup import job_table,job_skill_table


class JobPostings(Base):
    __table__ = job_table  # using the reflected table

    skills = relationship(
        "Skills",
        secondary=job_skill_table,
        back_populates="jobs"
    )