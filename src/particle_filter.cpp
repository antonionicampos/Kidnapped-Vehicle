/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;

static default_random_engine gen;

const int init_num_particles = 100;
const double init_weight = 1.0;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = init_num_particles;

	// create normal distribution for x, y and theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for(int i = 0; i < num_particles; i++) {
		Particle p = {i, dist_x(gen), dist_y(gen), dist_theta(gen), init_weight};
		particles.push_back(p);
		weights.push_back(init_weight);
	}

	// variables initialized
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen; // generates pseudo-random numbers

	// create noise for x, y and theta predictions
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	for(unsigned int i = 0; i < num_particles; i++) {

		if(fabs(yaw_rate) < 0.00001) {
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		} else {
			particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
			particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
			particles[i].theta += yaw_rate * delta_t;
		}

		// Add noise
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for(unsigned int i = 0; i < observations.size(); i++) {

		// current observation
		LandmarkObs observation = observations[i];
		
		// set minimum to maximum value possible for data type
		double min_distance = numeric_limits<double>::max();

		int index;

		for(unsigned int j = 0; j < predicted.size(); ++j) {

			// current prediction
			LandmarkObs prediction = predicted[j];

			double distance = dist(observation.x, observation.y, prediction.x, prediction.y);

			if(distance < min_distance) {
				min_distance = distance;
				index = prediction.id;
			}
		}

		observations[i].id = index;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for(unsigned int i = 0; i < num_particles; ++i) {

		// coordinates for current particle
		double x_particle = particles[i].x;
		double y_particle = particles[i].y;
		double theta_particle = particles[i].theta;

		// landmarks locations
		vector<LandmarkObs> predictions;

		for(unsigned int j = 0; j < map_landmarks.landmark_list.size(); ++j) {

			// coordinates and id for current map landmark
			double x_landmark = map_landmarks.landmark_list[j].x_f;
			double y_landmark = map_landmarks.landmark_list[j].y_f;
			int id_landmark = map_landmarks.landmark_list[j].id_i;

			// only landmarks inside sensor range
			if(fabs(x_landmark - x_particle) <= sensor_range && fabs(y_landmark - y_particle) <= sensor_range) {
				predictions.push_back(LandmarkObs{id_landmark, x_landmark, y_landmark});
			}

		}

		// Transform observations from vehicle coordinates to map coordinates
		vector<LandmarkObs> transformed_obs;
		for(unsigned int j = 0; j < observations.size(); ++j) {
			double x_t = x_particle + cos(theta_particle) * observations[j].x - sin(theta_particle) * observations[j].y;
			double y_t = y_particle + (sin(theta_particle) * observations[j].x) + (cos(theta_particle) * observations[j].y);
			transformed_obs.push_back(LandmarkObs{observations[j].id, x_t, y_t});
		}

		// Nearest neighbor
		dataAssociation(predictions, transformed_obs);

		particles[i].weight = 1.0;

		for(unsigned int j = 0; j < transformed_obs.size(); ++j) {
			// observation and prediction coordinates
			double x_obs, y_obs, x_pred, y_pred;
			x_obs = transformed_obs[j].x;
			y_obs = transformed_obs[j].y;

			int assoc_pred_id = transformed_obs[j].id;

			// x and y coordinates of the associated prediction
			for(unsigned int k = 0; k < predictions.size(); ++k) {

				if(predictions[k].id == assoc_pred_id) {
					x_pred = predictions[k].x;
					y_pred = predictions[k].y;
				}

			}

			// calculate weight
			// pre-computation
			double term1 = ((x_pred-x_obs)*(x_pred-x_obs)) / (2*std_landmark[0]*std_landmark[0]);
			double term2 = ((y_pred-y_obs)*(y_pred-y_obs)) / (2*std_landmark[1]*std_landmark[1]);;

			double w_obs = (1 / (2 * M_PI * std_landmark[0] * std_landmark[1])) * exp(-(term1 + term2));

			particles[i].weight *= w_obs;

		}

		weights[i] = particles[i].weight;

	}
	
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine gen; // generates pseudo-random numbers

	// get the weights
	vector<double> weights;
	for(unsigned int i = 0; i < num_particles; ++i) {
		weights.push_back(particles[i].weight);
	}

	// discrete_distribution
	vector<Particle> new_particles;
	discrete_distribution<int> index(weights.begin(), weights.end());
	for(unsigned int j = 0; j < num_particles; ++j) {
		const int i = index(gen);
		new_particles.push_back(particles[i]);
	}

	particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	// particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
