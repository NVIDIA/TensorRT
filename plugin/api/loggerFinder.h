#ifndef TRT_PLUGIN_API_LOGGER_FINDER_H
#define TRT_PLUGIN_API_LOGGER_FINDER_H

#include "plugin/common/vfcCommon.h"

namespace nvinfer1
{

namespace plugin
{
class VCPluginLoggerFinder : public ILoggerFinder
{
public:
    ILogger* findLogger() override
    {
        return getLogger();
    }
};

VCPluginLoggerFinder gVCPluginLoggerFinder;

//!
//! \brief Set a Logger finder for Version Compatibility (VC) plugin library so that all VC plugins can
//! use getLogger without dependency on nvinfer. This function shall be called once for the loaded vc plugin
//! library.
//!
//! \param setLoggerFinderFunc function in VC plugin library for setting logger finder.
//!
void setVCPluginLoggerFinder(std::function<void(ILoggerFinder*)> setLoggerFinderFunc)
{
    setLoggerFinderFunc(static_cast<ILoggerFinder*>(&gVCPluginLoggerFinder));
}

} // namespace plugin

} // namespace nvinfer1

#endif // TRT_RUNTIME_RT_LOGGER_FINDER_H
