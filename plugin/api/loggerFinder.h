#ifndef TRT_PLUGIN_API_LOGGER_FINDER_H
#define TRT_PLUGIN_API_LOGGER_FINDER_H

#include "common/vfcCommon.h"

namespace nvinfer1
{

namespace plugin
{
class VFCPluginLoggerFinder : public LoggerFinder
{
public:
    ILogger* findLogger() override
    {
        return getLogger();
    }
};

VFCPluginLoggerFinder gVFCPluginLoggerFinder;

//!
//! \brief Set a Logger finder for Version Forward Compatibility (VFC) plugin library so that all VFC plugins can
//! use getLogger without dependency on nvinfer. This function shall be called once for the loaded vfc plugin
//! library.
//!
void setVFCPluginLoggerFinder()
{
    setLoggerFinder(static_cast<LoggerFinder*>(&gVFCPluginLoggerFinder));
}

} // namespace plugin

} // namespace nvinfer1

#endif // TRT_RUNTIME_RT_LOGGER_FINDER_H
