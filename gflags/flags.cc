#include "flags.h"

static bool ValidateFile(const char* flagname, const std::string &file) {
   if (!file.empty())
     return true;
   printf("%s must be a valid filename.\n", flagname);
   return false;
}

static bool ValidateID(const char* flagname, int32_t id) {
   if (id >= 0)
     return true;
   printf("%s must >= 0\n", flagname);
   return false;
}

DEFINE_string(load_weights, "weights.bin", "weights file to load");
DEFINE_validator(load_weights, &ValidateFile);
DEFINE_string(save_weights, "weights.bin", "weights file to save");
DEFINE_validator(save_weights, &ValidateFile);
DEFINE_int32(cl_platform, 0, "opencl platform to use");
DEFINE_validator(cl_platform, &ValidateID);
DEFINE_int32(cl_device, 0, "opencl device to use");
DEFINE_validator(cl_device, &ValidateID);

void gflags_parse(int *argc, char ***argv)
{
    gflags::ParseCommandLineFlags(argc, argv, true);
}
