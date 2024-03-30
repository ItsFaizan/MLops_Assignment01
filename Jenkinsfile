pipeline {
    agent any
    
    stages {
        stage('Build Docker Image') {
            steps {
                script {
                    // Authenticate with Docker Hub
                    withCredentials([usernamePassword(credentialsId: 'docker-hub-credentials', usernameVariable: 'DOCKER_USERNAME', passwordVariable: 'DOCKER_PASSWORD')]) {
                        docker.withRegistry('https://index.docker.io/v1/', DOCKER_USERNAME, DOCKER_PASSWORD) {
                            // Build Docker image
                            def dockerImage = docker.build('itsfaizan71/mlops_a1:latest', '.')
                        }
                    }
                }
            }
        }
        stage('Push Docker Image') {
            steps {
                script {
                    // Authenticate with Docker Hub
                    withCredentials([usernamePassword(credentialsId: 'docker-hub-credentials', usernameVariable: 'DOCKER_USERNAME', passwordVariable: 'DOCKER_PASSWORD')]) {
                        docker.withRegistry('https://index.docker.io/v1/', DOCKER_USERNAME, DOCKER_PASSWORD) {
                            // Push Docker image to Docker Hub
                            dockerImage.push('latest')
                        }
                    }
                }
            }
        }
    }
}