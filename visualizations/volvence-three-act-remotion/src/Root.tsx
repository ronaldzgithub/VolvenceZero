import React from 'react';
import {Composition} from 'remotion';
import {
  FILM_DURATION_FRAMES,
  FILM_FPS,
  VolvenceThreeAct,
} from './ThreeActFilm';

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="VolvenceThreeAct"
      component={VolvenceThreeAct}
      durationInFrames={FILM_DURATION_FRAMES}
      fps={FILM_FPS}
      width={1920}
      height={1080}
      defaultProps={{}}
    />
  );
};

