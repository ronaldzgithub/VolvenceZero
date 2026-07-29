import React from 'react';
import {Composition} from 'remotion';
import {
  CORPORATE_FILM_DURATION,
  CORPORATE_FILM_FPS,
  VolvenceCorporateFilm,
} from './VolvenceCorporateFilm';

export const RemotionRoot: React.FC = () => (
  <Composition
    id="VolvenceCorporateFilm"
    component={VolvenceCorporateFilm}
    durationInFrames={CORPORATE_FILM_DURATION}
    fps={CORPORATE_FILM_FPS}
    width={1920}
    height={1080}
    defaultProps={{}}
  />
);
