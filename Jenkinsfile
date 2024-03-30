pipeline {
    agent any
    
    stages {
        stage('Checkout') {
            steps {
                // Checkout the code from the main branch
                git branch: 'main', url: 'https://github.com/ItsFaizan/MLops_Assignment01.git'
            }
        }
        
        stage('Build and Push Docker Image') {
            steps {
                script {
                    // Build Docker image
                    docker.build('your-docker-image:latest')
                    
                    // Authenticate with Docker Hub
                    docker.withRegistry('https://registry.hub.docker.com', 'docker-hub-credentials') {
                        // Push Docker image to Docker Hub
                        docker.image('your-docker-image:latest').push('latest')
                    }
                }
            }
        }
    }
    
    post {
        success {
            // Send success notification
            emailext subject: 'Docker image successfully built and pushed',
                      body: 'The Docker image for your application has been built and pushed to Docker Hub.',
                      to: 'faizanjavid71@.com'
        }
        failure {
            // Send failure notification
            emailext subject: 'Failed to build and push Docker image',
                      body: 'There was an error while building and pushing the Docker image to Docker Hub.',
                      to: 'faizanjavid71@.com'
        }
    }
}
