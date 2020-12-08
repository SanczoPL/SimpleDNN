
#ifndef STRUCTURES_H
#define STRUCTURES_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <QObject>
#include <QtCore>
#include <QColor>
#include <QDir>
#include <QString>
#include <QVector>
#include <QDebug>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>


#include "dnn.h"
#include "dlibnetwork.h"
#include "../ConfigReader/include/configreader.h"



constexpr auto COLORS{ "Colors" };
constexpr auto BLACK{ "Black" };
constexpr auto WHITE{ "White" };
constexpr auto SHADOW{ "Shadow" };
constexpr auto OUT_OF_SCOPE{ "OutOfScope" };
constexpr auto UNKNOWN{ "Unknown" };

constexpr auto ROI{ "ROI" };
constexpr auto GRAY_COLOR{ "GrayColor" };
constexpr auto R{ "R" };
constexpr auto G{ "G" };
constexpr auto B{ "B" };
constexpr auto A{ "A" };

constexpr auto P1X{ "p1x" };
constexpr auto P1Y{ "p1y" };
constexpr auto P2X{ "p2x" };
constexpr auto P2Y{ "p2y" };

constexpr auto PRE_PROCESSING{ "PreProcessing" };
constexpr auto POST_PROCESSING{ "PostProcessing" };
constexpr auto BACKGROUND{ "BackgroundSubtractor" };

constexpr auto GENERAL{ "General" };
constexpr auto LOG_LEVEL{ "LogLevel" };

constexpr auto DATASET{ "Dataset" };
constexpr auto DNN{ "DNN" };

constexpr auto INPUT_FOLDER{ "InputFolder" };
constexpr auto OUTPUT_FOLDER{ "OutputFolder" };
constexpr auto OUTPUT_FOLDER_GT{ "OutputFolderGT" };
constexpr auto OUTPUT_FOLDER_DIFF{ "OutputFolderDiff" };
constexpr auto OUTPUT_FOLDER_JSON{ "OutputFolderJSON" };

constexpr auto SETTING_AUTO_SAVE{ "SettingAutoSave" };

constexpr auto NOISE{ "Noise" };
constexpr auto FOLDER{ "FolderInput" };
constexpr auto STREAM_INPUT{ "StreamInput" };
constexpr auto VIDEO_GT{ "VideoGT" };
constexpr auto START_FRAME{ "StartFrame" };
constexpr auto STOP_FRAME{ "StopFrame" };
constexpr auto START_GT{ "StartGT" };
constexpr auto STOP_GT{ "StopGT" };
constexpr auto RESIZE{ "Resize" };

constexpr auto PATH_TO_DATASET{ "PathToDataset" };
constexpr auto CONFIG_NAME{ "ConfigName" };

constexpr auto INPUT_DATA{ "input" };
constexpr auto CLEAN{ "clean" };
constexpr auto GT_TEMP{ "gt_temp" };

constexpr auto GT{ "gt" };
constexpr auto TRACKER_GT{ "TrackerGT" };
constexpr auto TRACKER_ROI{ "TrackerROI" };
constexpr auto BACKGROUND_GT{ "BackgroundGT" };
constexpr auto BACKGROUND_TEMP{ "BackgroundTemp" };
constexpr auto BACKGROUND_ROI{ "BackgroundROI" };
constexpr auto INPUT_TYPE{ "InputType" };
constexpr auto OUTPUT_TYPE{ "OutputType" };
constexpr auto INPUT_PREFIX{ "InputPrefix" };

enum uiMode
{
  None = 0,
  SelectROI = 1,
  Paint = 2,
  MoveSelect = 3
};

struct images
{
  cv::Mat input;
  cv::Mat diff;
  cv::Mat noise;
  cv::Mat shadow;
};

struct listInfo
{
  qint32 id;
  QString label;
  qint32 size;
  bool enabled;
};

struct fileInfo
{
  QString filename;
  QString path;
  qint32 numberOfROI;
  QString infoROI;
  qint32 pixelWhite;
  qint32 pixelShadow;
  QString status;
};

struct colors
{
  QString name;
  QColor color;
  qint32 gray;
};

struct itemOnScene
{
  listInfo info;
};

QVector<QString> static scanAllImages(QString path)
{
  QVector<QString> temp;
  QDir directory(path);
  QStringList images = directory.entryList(QStringList() << "*.jpg" << "*.png" << "*.PNG" << "*.JPG", QDir::Files);

  foreach (QString filename, images)
  {
    QStringList sl = filename.split(".");
    temp.push_back(sl[0]);
  }
  return temp;
}
QVector<QString> static scanAllVideo(QString path)
{
    QVector<QString> temp;
    QDir directory(path);
    QStringList images = directory.entryList(QStringList() << "*.MP4" << "*.mp4", QDir::Files);

    foreach(QString filename, images)
    {
        QStringList sl = filename.split(".");
        temp.push_back(sl[0]);
    }
    return temp;

}



#endif // STRUCTURES_H
