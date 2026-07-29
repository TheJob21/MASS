import numpy as np

class Rewards():

    def computeRewardsForAgents(
        cognitiveAgents,
        binOwnership,
        config,
        startingFrequency
    ):
        """
        Generic reward computation for any agent group.

        Parameters
        ----------
        numAgents : int
            Number of agents in this group
        agentActionMap : dict[int, (start, stop)]
            Actions for this agent group
        agentPrevRewardMap : dict[int, float]
            Per-step rewards
        agentCumulativeRewardMap : dict[int, float]
            Accumulated rewards
        interferingActionMaps : list[dict]
            Other agents' action maps that cause interference
        """
        fftSize = config.FFT_SIZE
        binSize = config.BIN_SIZE
        channelBandwidth = config.CHANNEL_BANDWIDTH


        for cogAgent in cognitiveAgents:
            currAction = cogAgent.currentAction
            if currAction is None:
                continue

            raw_start, raw_stop = currAction
            raw_bw_bins = raw_stop - raw_start

            reward = 0.0
            if raw_bw_bins > 0:

                agent_id = cognitiveAgents.index(cogAgent) + 2

                if cogAgent.isTransmitting:
                    left_overflow = max(0, -raw_start)
                    right_overflow = max(0, raw_stop - fftSize)
                    overflow_bins = left_overflow + right_overflow
                    
                    exec_start = max(0, raw_start)
                    exec_stop = min(fftSize, raw_stop)
                    
                    ownership_slice = binOwnership[exec_start:exec_stop]

                    total_bins = exec_stop - exec_start
                    if config.MULTI_AGENT:
                        owned_mask = (ownership_slice == agent_id)
                        owned_bins = np.sum(owned_mask)

                        collision_bins = total_bins - owned_bins
                    else:
                        # Only static causes collision
                        collision_mask = (ownership_slice == 1)

                        collision_bins = np.sum(collision_mask)
                    # Add overflow as collision
                    collision_bins += overflow_bins

                    txFrac = (total_bins * binSize) / channelBandwidth
                    collisionFrac = (collision_bins * binSize) / channelBandwidth

                    cleanTxFrac = max(0.0, txFrac - collisionFrac)

                    # Store Collision amount
                    cogAgent.collisions.append(collision_bins * binSize)

                    rewardSpectrum = config.REWARD['transmission_weight'] * cleanTxFrac - config.REWARD['collision_weight'] * collisionFrac
                    
                    avgCenterFreq = cogAgent.getAveCenterFreqForCPI()
                    avgBW = cogAgent.getAveBwForCPI()
                    agentCenterFreq, agentBW = cogAgent.curActionAsCenterFreqBW(binSize, startingFrequency)
                    deltaBW = abs(agentBW - avgBW)
                    deltaCenterFreq = abs(agentCenterFreq - avgCenterFreq)
                    
                    rewardAdapt = (config.REWARD['bandwidth_distortion'] * deltaBW / channelBandwidth) + (config.REWARD['center_distortion'] * deltaCenterFreq / channelBandwidth)

                    reward = rewardSpectrum - rewardAdapt
                # else: # Listening but not transmitting
                #     left_overflow = max(0, -raw_start)
                #     right_overflow = max(0, raw_stop - fftSize)
                #     overflow_bins = left_overflow + right_overflow
                    
                #     exec_start = max(0, raw_start)
                #     exec_stop = min(fftSize, raw_stop)
                    
                #     ownership_slice = binOwnership[exec_start:exec_stop]

                #     total_bins = exec_stop - exec_start
                #     if config.MULTI_AGENT:
                #         owned_mask = (ownership_slice == agent_id)
                #         owned_bins = np.sum(owned_mask)

                #         collision_bins = total_bins - owned_bins
                #     else:
                #         # Only static causes collision
                #         collision_mask = (ownership_slice == 1)

                #         collision_bins = np.sum(collision_mask)
                #     # Add overflow as collision
                #     collision_bins += overflow_bins
                    
                #     collisionFrac = (collision_bins * binSize) / channelBandwidth
                    
                    
                    
                #     reward -= (collisionFrac * (config.REWARD['collision_weight'] / 30))

            cogAgent.storeReward(reward)