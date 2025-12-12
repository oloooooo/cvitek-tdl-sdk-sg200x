#include <stdio.h>
#include <stdlib.h>
#include "cvi_tdl.h"

static int ReleaseImage(VIDEO_FRAME_INFO_S *frame) {
  CVI_S32 ret = CVI_SUCCESS;
  if (frame->stVFrame.u64PhyAddr[0] != 0) {
    ret = CVI_SYS_IonFree(frame->stVFrame.u64PhyAddr[0], frame->stVFrame.pu8VirAddr[0]);
    frame->stVFrame.u64PhyAddr[0] = (CVI_U64)0;
    frame->stVFrame.u64PhyAddr[1] = (CVI_U64)0;
    frame->stVFrame.u64PhyAddr[2] = (CVI_U64)0;
    frame->stVFrame.pu8VirAddr[0] = NULL;
    frame->stVFrame.pu8VirAddr[1] = NULL;
    frame->stVFrame.pu8VirAddr[2] = NULL;
  }
  return ret;
}

static void PrintPoseResult(const cvtdl_object_t *obj_meta) {
  for (uint32_t i = 0; i < obj_meta->size; ++i) {
    const cvtdl_pedestrian_meta *pedestrian = obj_meta->info[i].pedestrian_properity;
    printf("person %u bbox=[%.1f, %.1f, %.1f, %.1f]\n", i, obj_meta->info[i].bbox.x1,
           obj_meta->info[i].bbox.y1, obj_meta->info[i].bbox.x2, obj_meta->info[i].bbox.y2);

    if (pedestrian == NULL) {
      printf("  no pedestrian metadata available\n");
      continue;
    }

    for (int kp = 0; kp < 17; ++kp) {
      printf("  keypoint[%2d]: (%.1f, %.1f) score=%.3f\n", kp, pedestrian->pose_17.x[kp],
             pedestrian->pose_17.y[kp], pedestrian->pose_17.score[kp]);
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    printf("Usage: %s <det_model_path> <pose_model_path> <image_path> [loop_count]\n", argv[0]);
    return -1;
  }

  const char *det_model_path = argv[1];
  const char *pose_model_path = argv[2];
  const char *image_path = argv[3];
  int loop_count = (argc >= 5) ? atoi(argv[4]) : 1;

  CVI_S32 ret = 0;
  cvitdl_handle_t tdl_handle = NULL;
  ret = CVI_TDL_CreateHandle(&tdl_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create tdl handle failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN, det_model_path);
  if (ret != CVI_SUCCESS) {
    printf("Open detection model failed with %#x!\n", ret);
    CVI_TDL_DestroyHandle(tdl_handle);
    return ret;
  }

  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_ALPHAPOSE, pose_model_path);
  if (ret != CVI_SUCCESS) {
    printf("Open pose model failed with %#x!\n", ret);
    CVI_TDL_DestroyHandle(tdl_handle);
    return ret;
  }

  VIDEO_FRAME_INFO_S frame;
  if (CVI_SUCCESS != CVI_TDL_LoadBinImage(image_path, &frame, PIXEL_FORMAT_RGB_888_PLANAR)) {
    printf("Load image %s failed.\n", image_path);
    CVI_TDL_DestroyHandle(tdl_handle);
    return -1;
  }

  printf("Image loaded: %ux%u\n", frame.stVFrame.u32Width, frame.stVFrame.u32Height);

  for (int i = 0; i < loop_count; i++) {
    cvtdl_object_t obj_meta = {0};

    ret = CVI_TDL_MobileDetV2_Pedestrian(tdl_handle, &frame, &obj_meta);
    if (ret != CVI_TDL_SUCCESS) {
      printf("Pedestrian detection failed with %#x!\n", ret);
      CVI_TDL_Free(&obj_meta);
      break;
    }

    printf("[%d] detected %u person(s)\n", i, obj_meta.size);

    ret = CVI_TDL_AlphaPose(tdl_handle, &frame, &obj_meta);
    if (ret != CVI_TDL_SUCCESS) {
      printf("AlphaPose failed with %#x!\n", ret);
      CVI_TDL_Free(&obj_meta);
      break;
    }

    PrintPoseResult(&obj_meta);
    CVI_TDL_Free(&obj_meta);
  }

  ReleaseImage(&frame);
  CVI_TDL_DestroyHandle(tdl_handle);
  return ret;
}
