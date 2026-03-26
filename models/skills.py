from sqlalchemy.orm import relationship
from DbIndexing.models.setup import Base,skills_table,job_skill_table


class Skills(Base):
    __table__ = skills_table

    jobs = relationship(
        "JobPostings",
        secondary=job_skill_table,
        back_populates="skills"
    )