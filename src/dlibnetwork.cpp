#include "dlibnetwork.h"

#define DEBUG TRUE
#define THREADS 4

DlibNetwork::DlibNetwork(QJsonObject const& a_config) {
	spdlog::trace("DlibNetwork::DlibNetwork()done");
#if(DEBUG)
	qDebug() << "DlibNetwork::DlibNetwork() config:" << a_config;
#endif
	m_det_minibatch_size = a_config["MinibatchSize"].toInt();
	m_imageSize = a_config["ImageSize"].toInt();
	m_queqe = a_config["Queqe"].toInt();
	m_dnnName = a_config["DnnName"].toString();
	m_synch_name = a_config["SynchName"].toString();
	m_useTwoCUDA = a_config["UseTwoCUDA"].toBool();

	//QString temp_m_learningRate = a_config["LearningRate"].toString();
	//spdlog::critical("DlibNetwork::DlibNetwork() m_learningRate:{}", temp_m_learningRate.toStdString());
	m_learningRate = a_config["LearningRate"].toDouble();
	m_inputType = ".png";

#if(DEBUG)
	spdlog::critical("DlibNetwork::DlibNetwork() m_imageSize:{}", m_imageSize);
	spdlog::critical("DlibNetwork::DlibNetwork() m_learningRate:{}", m_learningRate);
	int device_num = dlib::cuda::get_num_devices();
	std::cout << std::endl << "device_num():" << device_num;
	int device = dlib::cuda::get_device();
	std::cout << std::endl << "device():" << device;
	std::string device_name = dlib::cuda::get_device_name(0);
	std::cout << std::endl << "device_name():" << device_name;
#endif

#ifdef _WIN32
	m_split = "\\";
#endif // _WIN32
#ifdef __linux__
	m_split = "/";
#endif // _UNIX
}

DlibNetwork::~DlibNetwork() {}

void DlibNetwork::createData(QString gt, QString input)
{
	Logger->info("DlibNetwork::createData() gt:{}", gt.toStdString());
	Logger->info("DlibNetwork::createData() input:{}", input.toStdString());
	m_imageInfo.clear();
	QVector<QString> m_imgList = scanAllImages(gt);
	std::sort(m_imgList.begin(), m_imgList.end());
	Logger->info("DlibNetwork::createData() m_imgList:{}", m_imgList.size());
	if (m_imgList.size() > 0) {
		for (int iteration = 0; iteration < m_imgList.size(); iteration++) {
			if (iteration % 10 == 0)
			{
				spdlog::info("DlibNetwork::createData() iteration:{}", iteration);
			}
			QString name = input + m_split + m_imgList[iteration] + m_inputType;
			QString gtName = gt + m_split + m_imgList[iteration] + m_inputType;

				m_imageInfo.push_back({ name.toStdString(), gtName.toStdString() });
				spdlog::info("DlibNetwork::createData() m_imageInfo:({},{})", name.toStdString(), gtName.toStdString());
		}
	}
}

void DlibNetwork::trainDNN()
{
	std::cout << "m_imageInfo() ok... ";
	std::vector<truth_instance>  list;
	for (int i = 0; i < m_imageInfo.size(); i++)
	{
		list.push_back(load_truth_instances(m_imageInfo[i]));
	}
	Logger->trace("list.size():{}", list.size());

	net_type segb = train_segmentation_network(list, m_det_minibatch_size, m_imageSize, m_queqe, m_synch_name, m_useTwoCUDA, m_learningRate);

	std::cout << "Saving networks" << endl;
	QString dnnNameNet = m_dnnName;
	dlib::serialize(dnnNameNet.toStdString().c_str()) << segb;
}

truth_instance DlibNetwork::load_truth_instances(const image_info& info)
{
	dlib::matrix<unsigned char> input_image;
	dlib::matrix<unsigned char> label_image;

	dlib::load_image(input_image, info.image_filename);
	dlib::load_image(label_image, info.gt_filename);
	
	const auto nr = label_image.nr();
	const auto nc = label_image.nc();
	Logger->trace("DlibNetwork::load_truth_instances nr:{}, nc:{} ", nr,nc);

	dlib::matrix<float> result(nr, nc);

	for (long r = 0; r < nr; ++r)
	{
		for (long c = 0; c < nc; ++c)
		{
			const auto& index = label_image(r, c);
			if (index == 0)
				result(r, c) = -1;
			else
				result(r, c) = index / 255.0;
		}
	}
	truth_instance ti{ input_image , result };
	return ti;
}


net_type DlibNetwork::train_segmentation_network(
	const std::vector<truth_instance>& truth_images,
	unsigned int seg_minibatch_size,
	int imageSize,
	int m_que,
	QString a_synchronization_file_name,
	bool useTwoCUDA,
	double learningRate
)
{
	net_type seg_net;

	std::cout << "The net has " << seg_net.num_layers << " layers in it." << std::endl;
	std::cout << seg_net << std::endl;

	Logger->trace("truth_images[0].class_label_image.nr:{}", truth_images[0].input_image.nr());
	Logger->trace("truth_images[0].class_label_image.nc:{}", truth_images[0].input_image.nc());
	Logger->trace("truth_images[0].label_image.nr:{}", truth_images[0].label_image.nr());
	Logger->trace("truth_images[0].label_image.nc:{}", truth_images[0].label_image.nc());

	const double initial_learning_rate = 0.01;
	const double weight_decay = 0.0001;
	const double momentum = 0.9;

	const std::string synchronization_file_name = a_synchronization_file_name.toStdString();
	std::vector<int> cuda{ 0 };
	if (useTwoCUDA)
	{
		cuda.push_back(1);
	}

	dlib::dnn_trainer<net_type> seg_trainer(seg_net, dlib::sgd(weight_decay, momentum), cuda);
	seg_trainer.set_max_num_epochs(100);
	seg_trainer.be_verbose();
	seg_trainer.set_learning_rate(initial_learning_rate);
	seg_trainer.set_synchronization_file(synchronization_file_name, std::chrono::minutes(10));
	//seg_trainer.set_iterations_without_progress_threshold(2);
	//set_all_bn_running_stats_window_sizes(seg_net, 1000);

	std::cout << seg_trainer << endl;

	std::vector<dlib::matrix<unsigned char>> samples;
	std::vector<dlib::matrix<float>> labels;

	// Start a bunch of threads that read images from disk and pull out random crops.  It's
	// important to be sure to feed the GPU fast enough to keep it busy.  Using multiple
	// thread for this kind of data preparation helps us do that.  Each thread puts the
	// crops into the data queue.
	dlib::pipe<seg_training_sample> data(m_que);
	auto f = [&data, &truth_images, imageSize](time_t seed)
	{
		dlib::rand rnd(time(0) + seed);
		dlib::matrix<unsigned char> input_image;
		dlib::matrix<float> label_image;

		//matrix<unsigned char> rgb_label_chip;
		seg_training_sample temp;
		while (data.is_enabled())
		{
			const auto random_index = rnd.get_random_32bit_number() % truth_images.size();

			const auto& input_image = truth_images[random_index].input_image;
			const auto label_image = truth_images[random_index].label_image;

			dlib::matrix<unsigned char> inputDlib = truth_images[random_index].input_image;
			dlib::matrix<float> labelDlib = truth_images[random_index].label_image;

			cv::Mat inputMat = dlib::toMat(inputDlib);
			cv::Mat labelMat = dlib::toMat(labelDlib);


			const auto randomx = rnd.get_random_32bit_number() % (inputMat.cols - imageSize - 1);
			const auto randomy = rnd.get_random_32bit_number() % (inputMat.rows - imageSize - 1);

			cv::Rect randomRect(randomx, randomy, imageSize, imageSize);
			cv::Mat croppedInputMat = inputMat(randomRect);
			cv::Mat croppedLabelMat = labelMat(randomRect);

			dlib::cv_image<unsigned char> input_dlib_img(croppedInputMat);
			dlib::matrix<unsigned char> inputMatrixDlib;
			assign_image(inputMatrixDlib, input_dlib_img);

#if(DEBUG)

			//cv::imshow("label:", labelMat);
			//cv::imshow("croppedLabelMat:", croppedLabelMat);
			//cv::waitKey(0);
#endif

			dlib::cv_image<float> label_dlib_img(croppedLabelMat);
			dlib::matrix<float> labelMatrixDlib;
			assign_image(labelMatrixDlib, label_dlib_img);

			temp.input_image = inputMatrixDlib;
			temp.label_image = labelMatrixDlib;

			// Push the result to be used by the trainer.
			data.enqueue(temp);
		}
	};

	std::thread data_loader1([f]() { f(1); });
#if(THREADS >= 4)
	std::thread data_loader2([f]() { f(2); });
	std::thread data_loader3([f]() { f(3); });
	std::thread data_loader4([f]() { f(4); });
#endif
#if(THREADS >= 8)
	std::thread data_loader5([f]() { f(5); });
	std::thread data_loader6([f]() { f(6); });
	std::thread data_loader7([f]() { f(7); });
	std::thread data_loader8([f]() { f(8); });
#endif
#if(THREADS >= 16)
	std::thread data_loader9([f]() { f(9); });
	std::thread data_loader10([f]() { f(10); });
	std::thread data_loader11([f]() { f(11); });
	std::thread data_loader12([f]() { f(12); });
	std::thread data_loader13([f]() { f(13); });
	std::thread data_loader14([f]() { f(14); });
	std::thread data_loader15([f]() { f(15); });
	std::thread data_loader16([f]() { f(16); });
#endif
#if(THREADS >= 32)
	std::thread data_loader17([f]() { f(17); });
	std::thread data_loader18([f]() { f(18); });
	std::thread data_loader19([f]() { f(19); });
	std::thread data_loader20([f]() { f(20); });
	std::thread data_loader21([f]() { f(21); });
	std::thread data_loader22([f]() { f(22); });
	std::thread data_loader23([f]() { f(23); });
	std::thread data_loader24([f]() { f(24); });
	std::thread data_loader25([f]() { f(25); });
	std::thread data_loader26([f]() { f(26); });
	std::thread data_loader27([f]() { f(27); });
	std::thread data_loader28([f]() { f(28); });
	std::thread data_loader29([f]() { f(29); });
	std::thread data_loader30([f]() { f(30); });
	std::thread data_loader31([f]() { f(31); });
	std::thread data_loader32([f]() { f(32); });
#endif

	const auto stop_data_loaders = [&]()
	{
		data.disable();
		data_loader1.join();
#if(THREADS >= 4)
		data_loader2.join();
		data_loader3.join();
		data_loader4.join();
#endif
#if(THREADS >= 8)
		data_loader5.join();
		data_loader6.join();
		data_loader7.join();
		data_loader8.join();
#endif
#if(THREADS >= 16)
		data_loader9.join();
		data_loader10.join();
		data_loader11.join();
		data_loader12.join();
		data_loader13.join();
		data_loader14.join();
		data_loader15.join();
		data_loader16.join();
#endif
#if(THREADS >= 32)
		data_loader17.join();
		data_loader18.join();
		data_loader19.join();
		data_loader20.join();
		data_loader21.join();
		data_loader22.join();
		data_loader23.join();
		data_loader24.join();
		data_loader25.join();
		data_loader26.join();
		data_loader27.join();
		data_loader28.join();
		data_loader29.join();
		data_loader30.join();
		data_loader31.join();
		data_loader32.join();
#endif
	};

	Logger->info("start learning...");

	try
	{
		// The main training loop.  Keep making mini-batches and giving them to the trainer.
		// We will run until the learning rate has dropped by a factor of 1e-4.
		Logger->info("seg_trainer.get_learning_rate():{}", seg_trainer.get_learning_rate());
		while (seg_trainer.get_learning_rate() >= learningRate)
		{
			samples.clear();
			labels.clear();
			// make a mini-batch
			seg_training_sample temp;
			while (samples.size() < seg_minibatch_size)
			{
				data.dequeue(temp);
				samples.push_back(std::move(temp.input_image));
				labels.push_back(std::move(temp.label_image));
			}
			seg_trainer.train_one_step(samples, labels);
			//seg_trainer.train_one_step(samples.begin(), samples.end(), labels.begin());
		}
	}
	catch (std::exception&)
	{
		stop_data_loaders();
		seg_trainer.get_net();
		seg_net.clean();
		throw;
	}
	// Training done, tell threads to stop and make sure to wait for them to finish before moving on.
	stop_data_loaders();
	// also wait for threaded processing to stop in the trainer.
	seg_trainer.get_net();
	seg_net.clean();
	return seg_net;
}