#!groovy

def tryStep(String message, Closure block, Closure tearDown = null) {
    try {
        block();
    }
    catch (Throwable t) {
        slackSend message: "${env.JOB_NAME}: ${message} failure ${env.BUILD_URL}", channel: '#ci-channel', color: 'danger'

        throw t;
    }
    finally {
        if (tearDown) {
            tearDown();
        }
    }
}

node {
    stage("Checkout") {
        checkout scm
    }

    def image

    stage("Build develop image") {
        tryStep "build", {
            image = docker.build("repo.data.amsterdam.nl/datapunt/stadsarchief-ml:${env.BUILD_NUMBER}")
            image.push()
        }
    }

    stage("Test") {
        tryStep "Test", {
            image.inside {
                sh '/app/run_tests.sh'
                sh '/app/run_linting.sh'
            }
        }
    }
}

String BRANCH = "${env.BRANCH_NAME}"

if (BRANCH == "master" || BRANCH == "authentication") {
    node {
        stage('Push acceptance image') {
            tryStep "image tagging", {
                def image = docker.image("repo.data.amsterdam.nl/datapunt/stadsarchief-ml:${env.BUILD_NUMBER}")
                image.pull()
                image.push("acceptance")
            }
        }
    }

    stage('Waiting for approval') {
        slackSend channel: '#ci-channel', color: 'warning', message: 'Stadsarchief-ml is waiting for Production Release - please confirm'
        input "Deploy to Production?"
    }

    node {
        stage('Push production image') {
            tryStep "image tagging", {
                def image = docker.image("repo.data.amsterdam.nl/datapunt/stadsarchief-ml:${env.BUILD_NUMBER}")
                image.pull()
                image.push("production")
                image.push("latest")
            }
        }
    }
}
