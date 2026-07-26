import React from 'react';
import {Composition} from 'remotion';
import {
  FILM_DURATION_FRAMES,
  FILM_FPS,
  VolvenceThreeAct,
} from './ThreeActFilm';
import {
  PRODUCT_FOURTH_ACT_DURATION_FRAMES,
  PRODUCT_FOURTH_ACT_FPS,
  VolvenceProductFourthAct,
} from './ProductFourthActStandalone';
import {
  DEPLOYMENT_DURATION_FRAMES,
  DEPLOYMENT_FPS,
  VolvenceDeploymentStandalone,
} from './DeploymentStandalone';
import {
  DIGITAL_TWIN_DURATION_FRAMES,
  DIGITAL_TWIN_FPS,
  VolvenceDigitalTwinStandalone,
} from './DigitalTwinStandalone';
import {
  APP_ECOSYSTEM_DURATION_FRAMES,
  APP_ECOSYSTEM_FPS,
  VolvenceAppEcosystemStandalone,
} from './AppEcosystemStandalone';
import {
  HUMAN_MEANING_DURATION_FRAMES,
  HUMAN_MEANING_FPS,
  VolvenceHumanMeaningStandalone,
} from './HumanMeaningStandalone';

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="VolvenceThreeAct"
        component={VolvenceThreeAct}
        durationInFrames={FILM_DURATION_FRAMES}
        fps={FILM_FPS}
        width={1920}
        height={1080}
        defaultProps={{}}
      />
      <Composition
        id="VolvenceProductFourthAct"
        component={VolvenceProductFourthAct}
        durationInFrames={PRODUCT_FOURTH_ACT_DURATION_FRAMES}
        fps={PRODUCT_FOURTH_ACT_FPS}
        width={1920}
        height={1080}
        defaultProps={{}}
      />
      <Composition
        id="VolvenceDeploymentStandalone"
        component={VolvenceDeploymentStandalone}
        durationInFrames={DEPLOYMENT_DURATION_FRAMES}
        fps={DEPLOYMENT_FPS}
        width={1920}
        height={1080}
        defaultProps={{}}
      />
      <Composition
        id="VolvenceDigitalTwinStandalone"
        component={VolvenceDigitalTwinStandalone}
        durationInFrames={DIGITAL_TWIN_DURATION_FRAMES}
        fps={DIGITAL_TWIN_FPS}
        width={1920}
        height={1080}
        defaultProps={{}}
      />
      <Composition
        id="VolvenceAppEcosystemStandalone"
        component={VolvenceAppEcosystemStandalone}
        durationInFrames={APP_ECOSYSTEM_DURATION_FRAMES}
        fps={APP_ECOSYSTEM_FPS}
        width={1920}
        height={1080}
        defaultProps={{}}
      />
      <Composition
        id="VolvenceHumanMeaningStandalone"
        component={VolvenceHumanMeaningStandalone}
        durationInFrames={HUMAN_MEANING_DURATION_FRAMES}
        fps={HUMAN_MEANING_FPS}
        width={1920}
        height={1080}
        defaultProps={{}}
      />
    </>
  );
};
