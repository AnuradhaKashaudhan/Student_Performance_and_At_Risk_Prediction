FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "from pathlib import Path; required = ('models/model.joblib', 'models/metadata.json'); missing = [path for path in required if not Path(path).is_file()]; assert not missing, f'Missing model artifact(s): {missing}'"

EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
