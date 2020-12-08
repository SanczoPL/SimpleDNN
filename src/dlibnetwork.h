#pragma once

#include "structures.h"
#include "dnn.h"



struct image_info;

class DlibNetwork : public QObject {
	Q_OBJECT

public:
	DlibNetwork(QJsonObject const& a_config);
	~DlibNetwork();

	void createData(QString gt, QString input);
	void trainDNN();
	truth_instance load_truth_instances(const image_info& info);
	net_type train_segmentation_network(const std::vector<truth_instance>& truth_images, unsigned int seg_minibatch_size, int imageSize, int m_que, QString a_synchronization_file_name, bool useTwoCUDA, double learningRate);
	std::vector<image_info> get_imageInfo() { return m_imageInfo; }

private:
	int m_det_minibatch_size{};
	int m_imageSize{};
	int m_queqe{};
	QString m_dnnName;
	QString m_synch_name;
	bool m_useTwoCUDA{};
	QString m_split;
	QString m_inputType;
	double m_learningRate{};
	std::vector<image_info>  m_imageInfo;
};