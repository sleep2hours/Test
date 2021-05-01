#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include "opencv2/features2d.hpp"
#include <vector>
#include "../include/CameraCtl.hpp"

//Some constants for the algorithm
const double pi = 3.142;
const double cthr = 0.00001;
const double alpha = 0.09;
const double cT = 0.05;
const double covariance0 = 6.0;
const double cf = 0.5;
const double cfbar = 1.0 - cf;
const double temp_thr = 9.0 * covariance0 * covariance0;
const double prune = -alpha * cT;
const double alpha_bar = 1.0 - alpha;

//Structure used for saving various components for each pixel
struct gaussian
{
	double mean[3], covariance;
	double weight; // Represents the measure to which a particular component defines the pixel value
	gaussian *Next;
	gaussian *Previous;
} * ptr, *start, *rear, *g_temp, *save, *next, *previous, *nptr, *temp_ptr;

struct Node0
{
	gaussian *pixel_s;
	gaussian *pixel_r;
	int no_of_components;
	Node0 *Next;
} * N_ptr, *N_start, *N_rear;

struct Node1
{
	cv::Mat gauss;
	int no_of_comp;
	Node1 *Next;
} * N1_ptr, *N1_start, *N1_rear;

//Some function associated with the structure management
Node0 *Create_Node(double info1, double info2, double info3);
void Insert_End_Node(Node0 *np);
gaussian *Create_gaussian(double info1, double info2, double info3);

Node0 *Create_Node(double info1, double info2, double info3)
{
	N_ptr = new Node0;
	if (N_ptr != NULL)
	{
		N_ptr->Next = NULL;
		N_ptr->no_of_components = 1;
		N_ptr->pixel_s = N_ptr->pixel_r = Create_gaussian(info1, info2, info3);
	}
	return N_ptr;
}

gaussian *Create_gaussian(double info1, double info2, double info3)
{
	ptr = new gaussian;
	if (ptr != NULL)
	{
		ptr->mean[0] = info1;
		ptr->mean[1] = info2;
		ptr->mean[2] = info3;
		ptr->covariance = covariance0;
		ptr->weight = alpha;
		ptr->Next = NULL;
		ptr->Previous = NULL;
	}
	return ptr;
}

void Insert_End_Node(Node0 *np)
{
	if (N_start != NULL)
	{
		N_rear->Next = np;
		N_rear = np;
	}
	else
		N_start = N_rear = np;
}

void Insert_End_gaussian(gaussian *nptr)
{
	if (start != NULL)
	{
		rear->Next = nptr;
		nptr->Previous = rear;
		rear = nptr;
	}
	else
		start = rear = nptr;
}

gaussian *Delete_gaussian(gaussian *nptr)
{
	previous = nptr->Previous;
	next = nptr->Next;
	if (start != NULL)
	{
		if (nptr == start && nptr == rear)
		{
			start = rear = NULL;
			delete nptr;
		}
		else if (nptr == start)
		{
			next->Previous = NULL;
			start = next;
			delete nptr;
			nptr = start;
		}
		else if (nptr == rear)
		{
			previous->Next = NULL;
			rear = previous;
			delete nptr;
			nptr = rear;
		}
		else
		{
			previous->Next = next;
			next->Previous = previous;
			delete nptr;
			nptr = next;
		}
	}
	else
	{
		std::cout << "Underflow........";
		exit(0);
	}
	return nptr;
}

int main()
{
	int i, j, k;
	i = j = k = 0;

	// Declare matrices to store original and resultant binary image
	cv::Mat orig_img, bin_img;

	//Declare a VideoCapture object to store incoming frame and initialize it
	//cv::VideoCapture capture("/home/lym/PETS2009_sample_1.avi");
	cm::CameraCtl cap;
	int read = cap.startGrabbing();
	if (read)
	{
		std::cout << "Read video failed!" << std::endl;
		return 1;
	}
	orig_img = cap.getOpencvMat();
	//Recieveing input from the source and converting it to grayscale
	// capture.read(orig_img);
	if (orig_img.empty())
	{
		std::cout << "empty img!" << std::endl;
		return 1;
	}
	cv::cvtColor(orig_img, orig_img, CV_BGR2GRAY);

	//Initializing the binary image with the same dimensions as original image
	bin_img = cv::Mat(orig_img.rows, orig_img.cols, CV_8U, cv::Scalar(0));

	//Step 1: initializing with one gaussian for the first time and keeping the no. of models as 1
	uchar *r_ptr;
	uchar *b_ptr;
	for (i = 0; i < orig_img.rows; i++)
	{
		r_ptr = orig_img.ptr(i);
		for (j = 0; j < orig_img.cols; j++)
		{

			N_ptr = Create_Node(*r_ptr, *(r_ptr + 1), *(r_ptr + 2));
			if (N_ptr != NULL)
			{
				N_ptr->pixel_s->weight = 1.0;
				Insert_End_Node(N_ptr);
			}
			else
			{
				std::cout << "Memory limit reached... ";
				exit(0);
			}
		}
	}
	orig_img = cap.getOpencvMat();
	if (orig_img.empty())
	{
		std::cout << "empty img!" << std::endl;
		return 1;
	}
	int nL, nC;
	if (orig_img.isContinuous() == true)
	{
		nL = 1;
		nC = orig_img.rows * orig_img.cols * orig_img.channels();
	}
	else
	{
		nL = orig_img.rows;
		nC = orig_img.cols * orig_img.channels();
	}
	double mal_dist;
	double sum = 0.0;
	int count = 0;
	bool close = false;
	int background;
	double mult;
	double duration, duration1, duration2, duration3;
	double temp_cov = 0.0;
	double weight = 0.0;
	double var = 0.0;
	double muR, muG, muB, dR, dG, dB, rVal, gVal, bVal;

	//Step 2: Modelling each pixel with Gaussian
	duration1 = static_cast<double>(cv::getTickCount());
	while (1)
	{
		bin_img = cv::Mat(orig_img.rows, orig_img.cols, CV_8UC1, cv::Scalar(0));
		duration3 = 0.0;
		orig_img = cap.getOpencvMat();
		if (orig_img.empty())
		{
			std::cout << "empty img!" << std::endl;
			return 1;
		}
		int count = 0;
		N_ptr = N_start;
		duration = static_cast<double>(cv::getTickCount());
		for (i = 0; i < nL; i++)
		{
			r_ptr = orig_img.ptr(i);
			b_ptr = bin_img.ptr(i);
			for (j = 0; j < nC; j += 3)
			{
				sum = 0.0;
				close = false;
				background = 0;

				rVal = *(r_ptr++);
				gVal = *(r_ptr++);
				bVal = *(r_ptr++);

				start = N_ptr->pixel_s;
				rear = N_ptr->pixel_r;
				ptr = start;

				temp_ptr = NULL;

				if (N_ptr->no_of_components > 4)
				{
					Delete_gaussian(rear);
					N_ptr->no_of_components--;
				}

				for (k = 0; k < N_ptr->no_of_components; k++)
				{

					weight = ptr->weight;
					mult = alpha / weight;
					weight = weight * alpha_bar + prune;
					if (close == false)
					{
						muR = ptr->mean[0];
						muG = ptr->mean[1];
						muB = ptr->mean[2];

						dR = rVal - muR;
						dG = gVal - muG;
						dB = bVal - muB;
						var = ptr->covariance;

						mal_dist = (dR * dR + dG * dG + dB * dB);

						if ((sum < 0.5 * cfbar) && (mal_dist < 5.0 * var * var))
							background = 255;

						if (mal_dist < 5.0 * var * var)
						{
							weight += alpha;
							close = true;
							ptr->mean[0] = muR + mult * dR;
							ptr->mean[1] = muG + mult * dG;
							ptr->mean[2] = muB + mult * dB;
							temp_cov = var + mult * (mal_dist - var);
							ptr->covariance = temp_cov < 5.0 ? 5.0 : (temp_cov > 20.0 ? 20.0 : temp_cov);
							temp_ptr = ptr;
						}
					}

					if (weight < -prune)
					{
						ptr = Delete_gaussian(ptr);
						weight = 0;
						N_ptr->no_of_components--;
					}
					else
					{
						sum += weight;
						ptr->weight = weight;
					}

					ptr = ptr->Next;
				}

				if (close == false)
				{
					ptr = new gaussian;
					ptr->weight = alpha;
					ptr->mean[0] = rVal;
					ptr->mean[1] = gVal;
					ptr->mean[2] = bVal;
					ptr->covariance = covariance0;
					ptr->Next = NULL;
					ptr->Previous = NULL;
					if (start == NULL)
						start = rear = NULL;
					else
					{
						ptr->Previous = rear;
						rear->Next = ptr;
						rear = ptr;
					}
					temp_ptr = ptr;
					N_ptr->no_of_components++;
				}

				ptr = start;
				while (ptr != NULL)
				{
					ptr->weight /= sum;
					ptr = ptr->Next;
				}

				while (temp_ptr != NULL && temp_ptr->Previous != NULL)
				{
					if (temp_ptr->weight <= temp_ptr->Previous->weight)
						break;
					else
					{
						next = temp_ptr->Next;
						previous = temp_ptr->Previous;
						if (start == previous)
							start = temp_ptr;
						previous->Next = next;
						temp_ptr->Previous = previous->Previous;
						temp_ptr->Next = previous;
						if (previous->Previous != NULL)
							previous->Previous->Next = temp_ptr;
						if (next != NULL)
							next->Previous = previous;
						else
							rear = previous;
						previous->Previous = temp_ptr;
					}

					temp_ptr = temp_ptr->Previous;
				}
				N_ptr->pixel_s = start;
				N_ptr->pixel_r = rear;
				*b_ptr++ = background;
				N_ptr = N_ptr->Next;
			}
		}

		duration = static_cast<double>(cv::getTickCount()) - duration;
		duration /= cv::getTickFrequency();

		std::cout << "\n duration :" << duration;
		std::cout << "\n counts : " << count;
		cv::imshow("video", orig_img);
		cv::imshow("gp", bin_img);
		cv::waitKey(1);
		
	}

	duration1 = static_cast<double>(cv::getTickCount()) - duration1;
	duration1 /= cv::getTickFrequency();

	std::cout << "\n duration1 :" << duration1;

	return 0;
}
