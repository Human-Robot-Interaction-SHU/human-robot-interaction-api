# Use the official Python image from the Docker Hub
FROM python:3.12-alpine

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory inside the container
WORKDIR /code

# Install dependencies
COPY requirements.txt /code/
RUN pip install -r requirements.txt

# Copy the Django project files into the container
COPY . /code/

# Expose the port that Django runs on
EXPOSE 8000

# Run Django development server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]






# FROM --platform=$BUILDPLATFORM python:3.12-alpine AS builder
# EXPOSE 8000
# WORKDIR /app
# COPY requirements.txt /app
# RUN pip3 install -r requirements.txt --no-cache-dir
# COPY . /app
# ENTRYPOINT ["python3"]
# CMD ["manage.py", "runserver", "0.0.0.0:8000"]
#

# FROM builder as dev-envs
# RUN <<EOF
# apk update
# apk add git
# EOF
#
# RUN <<EOF
# addgroup -S docker
# adduser -S --shell /bin/bash --ingroup docker vscode
# EOF
# # install Docker tools (cli, buildx, compose)
# COPY --from=gloursdocker/docker / /
# CMD ["manage.py", "runserver", "0.0.0.0:8000"]