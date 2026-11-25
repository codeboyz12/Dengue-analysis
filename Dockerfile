# syntax=docker/dockerfile:1
# ภาพของ pipeline: ติดตั้งแพ็กเกจครั้งเดียว ข้อมูล/config/รายงานอยู่นอกภาพ (mount ผ่าน docker-compose.yml)
#   target base = ใช้งานจริง (ค่าตั้งต้น)   target dev = มี pytest + โฟลเดอร์ tests สำหรับรันเทส
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Asia/Bangkok

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# ผู้ใช้ไม่ใช่ root; ไฟล์ที่เขียนลง data/ reports/ ที่ mount มาจะเป็นของ uid นี้ (ปรับด้วย user: ใน compose ได้)
RUN useradd --create-home --uid 1000 app \
 && mkdir -p /app/data /app/input /app/reports /app/config \
 && chown -R app:app /app
COPY --chown=app:app config ./config
USER app

ENTRYPOINT ["python", "-m", "dengue_pipeline"]
CMD ["status"]


FROM base AS dev
USER root
RUN pip install pytest
COPY --chown=app:app tests ./tests
USER app
ENTRYPOINT ["python", "-m", "pytest"]
CMD ["-q"]
