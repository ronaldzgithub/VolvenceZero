import React from 'react';
import {Composition} from 'remotion';
import {CognitiveAgiFilm} from './CognitiveAgiFilm';
import {
  ARCHITECTURE_DEEP_DIVE_DURATION_FRAMES,
  ARCHITECTURE_DEEP_DIVE_FPS,
  VolvenceArchitectureDeepDive,
  VolvenceArchitectureDeepDiveNarrated,
} from './VolvenceArchitectureDeepDive';
import {
  ContinuousLearningMechanism,
  CONTINUOUS_LEARNING_DURATION_FRAMES,
  CONTINUOUS_LEARNING_FPS,
} from './ContinuousLearningMechanism';
import {DialogueCaseMechanism} from './DialogueCaseMechanism';
import {
  COGNITIVE_AGI_DURATION_FRAMES,
  COGNITIVE_AGI_FPS,
} from './data/cognitiveAgi';
import {
  AUDIO_DURATION_SECONDS,
  AUDIO_START_SECONDS,
  OUTRO_DURATION_SECONDS,
} from './data/dialogueCase';

export const VIDEO_FPS = 30;
export const VIDEO_DURATION_FRAMES = Math.ceil(
  (AUDIO_START_SECONDS + AUDIO_DURATION_SECONDS + OUTRO_DURATION_SECONDS) *
    VIDEO_FPS,
);

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="VolvenceArchitectureDeepDive"
        component={VolvenceArchitectureDeepDive}
        durationInFrames={ARCHITECTURE_DEEP_DIVE_DURATION_FRAMES}
        fps={ARCHITECTURE_DEEP_DIVE_FPS}
        width={1920}
        height={1080}
        defaultProps={{}}
      />
      <Composition
        id="VolvenceArchitectureDeepDiveNarrated"
        component={VolvenceArchitectureDeepDiveNarrated}
        durationInFrames={ARCHITECTURE_DEEP_DIVE_DURATION_FRAMES}
        fps={ARCHITECTURE_DEEP_DIVE_FPS}
        width={1920}
        height={1080}
        defaultProps={{}}
      />
      <Composition
        id="ContinuousLearningMechanism"
        component={ContinuousLearningMechanism}
        durationInFrames={CONTINUOUS_LEARNING_DURATION_FRAMES}
        fps={CONTINUOUS_LEARNING_FPS}
        width={1920}
        height={1080}
        defaultProps={{}}
      />
      <Composition
        id="CognitiveAGI"
        component={CognitiveAgiFilm}
        durationInFrames={COGNITIVE_AGI_DURATION_FRAMES}
        fps={COGNITIVE_AGI_FPS}
        width={1920}
        height={1080}
        defaultProps={{}}
      />
      <Composition
        id="VolvenceCoreMechanism"
        component={DialogueCaseMechanism}
        durationInFrames={VIDEO_DURATION_FRAMES}
        fps={VIDEO_FPS}
        width={1920}
        height={1080}
        defaultProps={{}}
      />
    </>
  );
};
