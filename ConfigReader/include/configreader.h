#ifndef CONFIG_READER_H
#define CONFIG_READER_H

#include "../../spdlog/include/spdlog/spdlog.h"
#include "../../spdlog/include/spdlog/cfg/env.h"
#include "../../spdlog/include/spdlog/sinks/stdout_color_sinks.h"

extern std::shared_ptr<spdlog::logger> Logger;
extern std::shared_ptr<spdlog::logger> ErrorLogger;

#include <stdio.h>
#include <QJsonObject>
#include <QFile>
#include <QJsonDocument>

class ConfigReader : public QObject
{
 public:
  ConfigReader();
  ~ConfigReader();
  bool readConfig(QString configPathWithName, QJsonObject & obj);
};
  #endif //CONFIG_READER_H