#include "../include/configreader.h"

std::shared_ptr<spdlog::logger> Logger{ spdlog::stdout_color_mt("console") };
std::shared_ptr<spdlog::logger> ErrorLogger{ spdlog::stderr_color_mt("stderr") };

ConfigReader::ConfigReader() {
	Logger->trace("ConfigReader::ConfigReader()");
}

ConfigReader::~ConfigReader() {}

bool ConfigReader::readConfig(QString configPathWithName, QJsonObject& obj) {
	Logger->trace("ConfigReader::readConfig() try to load {}", (configPathWithName).toStdString());
	QFile jConfigFile{ configPathWithName };
	if (!jConfigFile.open(QIODevice::ReadOnly)) {
		Logger->error("Can not open json file:{}", (configPathWithName).toStdString());
		return false;
	}

	QJsonDocument jConfigDoc{ QJsonDocument::fromJson((jConfigFile.readAll())) };
	if (!jConfigDoc.isObject()) {
		Logger->error("Invalid json config file:{}", (configPathWithName).toStdString());
		return false;
	}
	obj = { jConfigDoc.object() };
	Logger->debug("{} load successful", (configPathWithName).toStdString());
	return true;
}
