#include "structures.h"

#include <iostream>
#include <iterator>
#include <thread>
#include <stdlib.h>
#include <stdio.h>

constexpr auto DATASET_UNIX{ "DatasetUnix" };
constexpr auto DATASET_WIN{ "DatasetWin" };

void intro();

int main(int argc, char** argv) try
{
	QJsonObject m_preprocess;
	QJsonObject m_config;
	QJsonObject m_dataset;

	Logger->set_pattern("[%Y-%m-%d] [%H:%M:%S.%e] [%t] [%^%l%$] %v");
	QString configPathWithName = "config.json";
	if (argc > 1)
	{
		configPathWithName = QString::fromStdString(argv[1]);
	}
	Logger->trace("MainLoop::MainLoop() open config file:{}", configPathWithName.toStdString());
	ConfigReader* configReader = new ConfigReader();
	if (!configReader->readConfig(configPathWithName, m_config)) {
		Logger->error("File {} not readed", configPathWithName.toStdString());
		return -66;
	}
	delete configReader;

	Logger->set_level(static_cast<spdlog::level::level_enum>(m_config[GENERAL].toObject()["LogLevel"].toInt()));

#ifdef _WIN32
	QJsonObject jDataset{ m_config[DATASET_WIN].toObject() };
#endif // _WIN32
#ifdef __linux__
	QJsonObject jDataset{ m_config[DATASET_UNIX].toObject() };
#endif // _UNIX

	QString input = jDataset["input"].toString();
	QString gt_temp = jDataset["gt_temp"].toString();
	DlibNetwork m_network{ m_config["DNN"].toObject() };
	m_network.createData(gt_temp,input);
	m_network.trainDNN();

}
catch (std::exception& e)
{
	std::cout << e.what() << std::endl;
}

void intro() {
	spdlog::info("\n\n\t\033[1;31mGenetic Algorithm\033[0m\n"
		"\t Author: Grzegorz Matczak\n"
		"\t 08.12.2020\n"
		"\t Simple DNN v1.0\n");
}