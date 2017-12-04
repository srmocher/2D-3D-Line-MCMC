import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# common camera matrix for all scenarios
camera_matrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1]])
num_iterations = 50000


def get_2d_projection_of_3d_point(point3d):
    projection = camera_matrix.dot(np.append(point3d,1))
    point_2d = projection[0:2]/projection[2]
    return point_2d

def get_log_likelihood(r_s,q_i,q_f,t_ratios,sigma_2d):
    num_points = r_s.shape[0]
    q_s = np.zeros(shape=(num_points, 2))
    for i in range(0, num_points):
        q_s[i] = q_i + (q_f - q_i) * t_ratios[i]

    log_likelihood = np.zeros(shape=(num_points, 2))
    for i in range(0, num_points):
        log_likelihood[i] = multivariate_normal.logpdf(r_points[i], q_s[i], sigma_2d)

    return log_likelihood

def get_prior(p_i,p_f,mu_3d,sigma_3d):
    return multivariate_normal.logpdf([p_i,p_f],mu_3d,sigma_3d)

def proposal(previous,sigma):
    return multivariate_normal.rvs(previous,sigma)

def plot_parameter(x,y,x_label,y_label,title):
    figure = plt.figure()

    figure.suptitle(title)
    plt.plot(x,y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)



def mh_sample_posterior_3d(t_ratios,r_points,sigma_2d,mu_3d,sigma_3d,p_i=None,p_f=None):


   q_i = get_2d_projection_of_3d_point(p_i)
   q_f = get_2d_projection_of_3d_point(p_f)


   log_prior = get_prior(p_i,p_f,mu_3d,sigma_3d)

   log_likelihood = get_log_likelihood(r_points,q_i,q_f,t_ratios,sigma_2d)

   log_posterior = np.sum(log_likelihood) + np.sum(log_prior)
   p_i_next = proposal(p_i,sigma_3d)
   p_f_next = proposal(p_f,sigma_3d)

   q_i_next = get_2d_projection_of_3d_point(p_i_next)
   q_f_next = get_2d_projection_of_3d_point(p_f_next)

   log_likelihood_next = get_log_likelihood(r_points,q_i_next,q_f_next,t_ratios,sigma_2d)

   log_prior_next = get_prior(p_i_next,p_f_next,mu_3d,sigma_3d)

   log_posterior_next = np.sum(log_likelihood_next) + np.sum(log_prior_next)

   log_r = log_posterior_next - log_posterior


   if log_r >=0:
       return p_i_next,p_f_next,log_posterior_next,True
   else:
       u = np.random.uniform()
       if log_r >= np.log(u):
           return p_i_next,p_f_next,log_posterior_next,True
       else:
           return p_i,p_f,log_posterior,False





r_points = np.genfromtxt('data/points_2d_camera_1.csv',delimiter=',')
t_ratios = np.genfromtxt('data/inputs.csv',delimiter=',')
sigma_3d = 6*np.identity(3)
mu_3d = np.array([0,0,4]).T
sigma_2d = (0.05)*(0.05)*np.identity(2)
pi_points = []
pf_points = []

p_i_current,p_f_current = multivariate_normal.rvs(mu_3d,sigma_3d,size=2)
posterior_probs = np.zeros(shape=(num_iterations,1))
for i in range(0,num_iterations):
    new_pi,new_pf,posterior,accepted = mh_sample_posterior_3d(t_ratios,r_points,sigma_2d,mu_3d,sigma_3d,p_i_current,p_f_current)
    posterior_probs[i] = posterior
    pi_points.append(new_pi)
    pf_points.append(new_pf)
    if accepted:
        p_i_current = new_pi
        p_f_current = new_pf

# Plot parameters
indices = np.linspace(0,num_iterations,num_iterations)
p_i_x = np.array(pi_points)[:,0]
p_i_y = np.array(pi_points)[:,1]
p_i_z = np.array(pi_points)[:,2]

p_f_x = np.array(pf_points)[:,0]
p_f_y = np.array(pf_points)[:,1]
p_f_z = np.array(pf_points)[:,2]

max_index_posterior = np.argmax(posterior_probs)
p_i_map_estimate = pi_points[max_index_posterior]
p_f_map_estimate = pf_points[max_index_posterior]

q_i_map_estimate = camera_matrix.dot(np.append(p_i_map_estimate,1))
q_f_map_estimate = camera_matrix.dot(np.append(p_f_map_estimate,1))

q_i_map_estimate = q_i_map_estimate[0:2]/q_i_map_estimate[2]
q_f_map_estimate = q_f_map_estimate[0:2]/q_f_map_estimate[2]
r_x = r_points[:,0]
r_y = r_points[:,1]
figure = plt.figure()
plt.scatter(r_x,r_y)
q_x = np.array([q_i_map_estimate[0],q_f_map_estimate[0]])
q_y = np.array([q_i_map_estimate[1],q_f_map_estimate[1]])
plt.plot(q_x,q_y)

plot_parameter(indices,p_i_x,"No of samples",r"$P_i$ x-coordinate",r"Variation of parameter $P_i(x)$")
plot_parameter(indices,p_i_y,"No of samples",r"$P_i$ y-coordinate","Variation of parameter")
plot_parameter(indices,p_i_z,"No of samples",r"$P_i$ z-coordinate","Variation of parameter")

plot_parameter(indices,p_f_x,"No of samples",r"$P_f$ x-coordinate","Variation of parameter")
plot_parameter(indices,p_f_y,"No of samples",r"$P_f$ y-coordinate","Variation of parameter")
plot_parameter(indices,p_f_z,"No of samples",r"$P_f$ z-coordinate","Variation of parameter")
print("Done")


