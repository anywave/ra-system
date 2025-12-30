{-|
Module      : Ra.GenomicHarmonics
Description : DNA-linked scalar keys for epigenetic modulation
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Maps emergence fields to symbolic representations of gene expression and
epigenetic switches, allowing harmonic signatures to influence epigenetic
field states.

== Genomic Harmonics Theory

=== DNA Resonance

DNA exhibits resonance properties:

* 528 Hz - DNA repair frequency
* Codon triplets map to harmonic triads
* Epigenetic marks respond to field coherence
* Telomere length correlates with scalar exposure

=== Harmonic-Gene Mapping

Spherical harmonic signatures map to gene clusters:

* (l=0, m=0) - Core metabolic genes
* (l=1, m=±1) - Stress response genes
* (l=2, m=0) - Growth/repair genes
* (l=2, m=±2) - Immune function genes

=== Epigenetic Modulation

Scalar fields influence epigenetics via:

* Methylation pattern shifts
* Histone modification cascades
* Non-coding RNA activation
* Chromatin remodeling
-}
module Ra.GenomicHarmonics
  ( -- * Core Types
    EpigeneticField(..)
  , GeneCluster(..)
  , HarmonicKey(..)
  , mkEpigeneticField

    -- * Harmonic Mapping
  , harmonicToCluster
  , clusterToHarmonic
  , mappingTable

    -- * Gene Clusters
  , ClusterState(..)
  , clusterActivation
  , clusterDescription

    -- * Field Modulation
  , FieldShift(..)
  , shiftFromEmergence
  , applyShift
  , cumulativeShift

    -- * Epigenetic Timeline
  , EpigeneticArc(..)
  , ArcPhase(..)
  , generateArc
  , arcProgress
  , arcPotential

    -- * Session Integration
  , SessionEpigenetics(..)
  , initSessionEpigenetics
  , updateFromEmergence
  , sessionTimeline
  ) where

import Ra.Constants.Extended
  ( phi, coherenceFloorPOR )

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Epigenetic field state
data EpigeneticField = EpigeneticField
  { efActiveClusters  :: ![GeneCluster]   -- ^ Currently active clusters
  , efFieldStrength   :: !Double          -- ^ Overall field strength [0,1]
  , efCoherence       :: !Double          -- ^ Field coherence [0,1]
  , efPhase           :: !Double          -- ^ Current phase [0, 2*pi]
  , efHistory         :: ![FieldShift]    -- ^ Shift history
  } deriving (Eq, Show)

-- | Gene cluster type
data GeneCluster
  = CoreMetabolic       -- ^ (0,0) - Basic metabolism
  | StressResponse      -- ^ (1,±1) - Stress/adaptation
  | GrowthRepair        -- ^ (2,0) - Growth and repair
  | ImmuneFunction      -- ^ (2,±2) - Immune system
  | NeuralPlasticity    -- ^ (3,0) - Brain plasticity
  | Longevity           -- ^ (3,±1) - Aging/telomeres
  | Circadian           -- ^ (4,0) - Circadian rhythm
  | Emotional           -- ^ (4,±2) - Emotional regulation
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Harmonic key for gene mapping
data HarmonicKey = HarmonicKey
  { hkL      :: !Int      -- ^ Harmonic degree
  , hkM      :: !Int      -- ^ Harmonic order
  , hkPhi    :: !Int      -- ^ Phi exponent (φ^n timing)
  } deriving (Eq, Show)

-- | Create epigenetic field
mkEpigeneticField :: Double -> EpigeneticField
mkEpigeneticField coherence = EpigeneticField
  { efActiveClusters = [CoreMetabolic]  -- Always active
  , efFieldStrength = coherence * 0.5
  , efCoherence = clamp01 coherence
  , efPhase = 0.0
  , efHistory = []
  }

-- =============================================================================
-- Harmonic Mapping
-- =============================================================================

-- | Map harmonic signature to gene cluster
harmonicToCluster :: (Int, Int) -> Maybe GeneCluster
harmonicToCluster (l, m) = case (l, m) of
  (0, 0)  -> Just CoreMetabolic
  (1, 1)  -> Just StressResponse
  (1, -1) -> Just StressResponse
  (2, 0)  -> Just GrowthRepair
  (2, 2)  -> Just ImmuneFunction
  (2, -2) -> Just ImmuneFunction
  (3, 0)  -> Just NeuralPlasticity
  (3, 1)  -> Just Longevity
  (3, -1) -> Just Longevity
  (4, 0)  -> Just Circadian
  (4, 2)  -> Just Emotional
  (4, -2) -> Just Emotional
  _       -> Nothing

-- | Map gene cluster to primary harmonic
clusterToHarmonic :: GeneCluster -> (Int, Int)
clusterToHarmonic cluster = case cluster of
  CoreMetabolic    -> (0, 0)
  StressResponse   -> (1, 1)
  GrowthRepair     -> (2, 0)
  ImmuneFunction   -> (2, 2)
  NeuralPlasticity -> (3, 0)
  Longevity        -> (3, 1)
  Circadian        -> (4, 0)
  Emotional        -> (4, 2)

-- | Full mapping table
mappingTable :: [(HarmonicKey, GeneCluster, String)]
mappingTable =
  [ (HarmonicKey 0 0 0, CoreMetabolic, "ATP synthesis, basic metabolism")
  , (HarmonicKey 1 1 1, StressResponse, "Cortisol regulation, HPA axis")
  , (HarmonicKey 1 (-1) 1, StressResponse, "Heat shock proteins, cellular stress")
  , (HarmonicKey 2 0 2, GrowthRepair, "Growth factors, tissue repair")
  , (HarmonicKey 2 2 2, ImmuneFunction, "Cytokine production, immune activation")
  , (HarmonicKey 2 (-2) 2, ImmuneFunction, "Inflammatory regulation")
  , (HarmonicKey 3 0 3, NeuralPlasticity, "BDNF, synaptic plasticity")
  , (HarmonicKey 3 1 3, Longevity, "Telomerase, SIRT genes")
  , (HarmonicKey 3 (-1) 3, Longevity, "Autophagy, cellular renewal")
  , (HarmonicKey 4 0 4, Circadian, "Clock genes, melatonin")
  , (HarmonicKey 4 2 4, Emotional, "Serotonin, dopamine pathways")
  , (HarmonicKey 4 (-2) 4, Emotional, "Oxytocin, bonding hormones")
  ]

-- =============================================================================
-- Gene Clusters
-- =============================================================================

-- | Cluster activation state
data ClusterState = ClusterState
  { csCluster     :: !GeneCluster
  , csActivation  :: !Double        -- ^ Activation level [0,1]
  , csExpression  :: !Double        -- ^ Expression rate [0,1]
  , csMethylation :: !Double        -- ^ Methylation level [0,1]
  , csStability   :: !Double        -- ^ State stability [0,1]
  } deriving (Eq, Show)

-- | Calculate cluster activation from field
clusterActivation :: GeneCluster -> EpigeneticField -> ClusterState
clusterActivation cluster field =
  let -- Base activation from field strength
      baseActivation = efFieldStrength field

      -- Bonus if cluster is in active list
      activeBonus = if cluster `elem` efActiveClusters field then 0.2 else 0.0

      -- Coherence modifier
      cohMod = efCoherence field * phi * 0.1

      activation = clamp01 (baseActivation + activeBonus + cohMod)

      -- Expression follows activation with delay
      expression = activation * 0.8

      -- Methylation inversely related to activation
      methylation = 1.0 - activation * 0.5

      -- Stability from coherence
      stability = efCoherence field
  in ClusterState
      { csCluster = cluster
      , csActivation = activation
      , csExpression = expression
      , csMethylation = methylation
      , csStability = stability
      }

-- | Get cluster description
clusterDescription :: GeneCluster -> String
clusterDescription cluster = case cluster of
  CoreMetabolic -> "Core metabolic processes and energy production"
  StressResponse -> "Stress adaptation and resilience pathways"
  GrowthRepair -> "Tissue growth, healing, and regeneration"
  ImmuneFunction -> "Immune system activation and regulation"
  NeuralPlasticity -> "Brain plasticity and cognitive function"
  Longevity -> "Cellular aging and longevity factors"
  Circadian -> "Circadian rhythm and sleep regulation"
  Emotional -> "Emotional regulation and social bonding"

-- =============================================================================
-- Field Modulation
-- =============================================================================

-- | Field shift from emergence
data FieldShift = FieldShift
  { fsTargetCluster :: !GeneCluster
  , fsMagnitude     :: !Double       -- ^ Shift magnitude [-1, 1]
  , fsDirection     :: !ShiftDir
  , fsTimestamp     :: !Double       -- ^ Session time
  } deriving (Eq, Show)

-- | Shift direction
data ShiftDir = Upregulate | Downregulate | Neutral
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Generate shift from emergence event
shiftFromEmergence :: Double -> (Int, Int) -> Double -> FieldShift
shiftFromEmergence alpha (l, m) timestamp =
  let cluster = case harmonicToCluster (l, m) of
        Just c -> c
        Nothing -> CoreMetabolic

      -- Magnitude from alpha
      magnitude = alpha * phi * 0.5

      -- Direction from alpha threshold
      direction = if alpha > coherenceFloorPOR
                  then Upregulate
                  else if alpha < 0.3
                  then Downregulate
                  else Neutral
  in FieldShift
      { fsTargetCluster = cluster
      , fsMagnitude = magnitude
      , fsDirection = direction
      , fsTimestamp = timestamp
      }

-- | Apply shift to epigenetic field
applyShift :: FieldShift -> EpigeneticField -> EpigeneticField
applyShift shift field =
  let -- Modify field strength
      strengthDelta = case fsDirection shift of
        Upregulate -> fsMagnitude shift * 0.1
        Downregulate -> -(fsMagnitude shift * 0.1)
        Neutral -> 0.0

      newStrength = clamp01 (efFieldStrength field + strengthDelta)

      -- Add cluster if upregulating
      newClusters = if fsDirection shift == Upregulate &&
                       fsTargetCluster shift `notElem` efActiveClusters field
                    then fsTargetCluster shift : efActiveClusters field
                    else efActiveClusters field

      -- Update history
      newHistory = shift : take 100 (efHistory field)
  in field
      { efFieldStrength = newStrength
      , efActiveClusters = newClusters
      , efHistory = newHistory
      }

-- | Calculate cumulative shift
cumulativeShift :: EpigeneticField -> Double
cumulativeShift field =
  let shifts = efHistory field
      magnitudes = map fsMagnitude shifts
  in sum magnitudes / fromIntegral (max 1 (length magnitudes))

-- =============================================================================
-- Epigenetic Timeline
-- =============================================================================

-- | Epigenetic arc over session
data EpigeneticArc = EpigeneticArc
  { eaCluster    :: !GeneCluster
  , eaStartLevel :: !Double       -- ^ Starting activation
  , eaEndLevel   :: !Double       -- ^ Target activation
  , eaPhase      :: !ArcPhase
  , eaDuration   :: !Double       -- ^ Arc duration (seconds)
  , eaProgress   :: !Double       -- ^ Current progress [0,1]
  } deriving (Eq, Show)

-- | Arc phase
data ArcPhase
  = ArcInitiation   -- ^ Beginning of change
  | ArcBuilding     -- ^ Change accumulating
  | ArcPeak         -- ^ Maximum effect
  | ArcIntegration  -- ^ Stabilizing new state
  | ArcComplete     -- ^ Change complete
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Generate arc from field shifts
generateArc :: GeneCluster -> [FieldShift] -> EpigeneticArc
generateArc cluster shifts =
  let relevantShifts = filter (\s -> fsTargetCluster s == cluster) shifts

      -- Calculate start and end from shifts
      startLevel = 0.3  -- Baseline
      totalMag = sum (map fsMagnitude relevantShifts)
      endLevel = clamp01 (startLevel + totalMag * 0.5)

      -- Duration from phi relationship
      duration = phi ** fromIntegral (length relevantShifts) * 60.0

      -- Phase from progress
      progress = if null relevantShifts then 0.0 else 0.5
      phase = progressToPhase progress
  in EpigeneticArc
      { eaCluster = cluster
      , eaStartLevel = startLevel
      , eaEndLevel = endLevel
      , eaPhase = phase
      , eaDuration = duration
      , eaProgress = progress
      }

-- Progress to phase
progressToPhase :: Double -> ArcPhase
progressToPhase p
  | p < 0.2 = ArcInitiation
  | p < 0.4 = ArcBuilding
  | p < 0.6 = ArcPeak
  | p < 0.8 = ArcIntegration
  | otherwise = ArcComplete

-- | Get arc progress
arcProgress :: EpigeneticArc -> Double
arcProgress = eaProgress

-- | Get arc potential (how much change possible)
arcPotential :: EpigeneticArc -> Double
arcPotential arc = abs (eaEndLevel arc - eaStartLevel arc)

-- =============================================================================
-- Session Integration
-- =============================================================================

-- | Session epigenetics state
data SessionEpigenetics = SessionEpigenetics
  { seField     :: !EpigeneticField
  , seArcs      :: ![EpigeneticArc]
  , seDuration  :: !Double            -- ^ Session duration so far
  , seEvents    :: !Int               -- ^ Number of emergence events
  } deriving (Eq, Show)

-- | Initialize session epigenetics
initSessionEpigenetics :: Double -> SessionEpigenetics
initSessionEpigenetics coherence = SessionEpigenetics
  { seField = mkEpigeneticField coherence
  , seArcs = []
  , seDuration = 0.0
  , seEvents = 0
  }

-- | Update from emergence event
updateFromEmergence :: Double -> (Int, Int) -> Double -> SessionEpigenetics -> SessionEpigenetics
updateFromEmergence alpha harmonic dt session =
  let -- Generate shift
      shift = shiftFromEmergence alpha harmonic (seDuration session)

      -- Apply to field
      newField = applyShift shift (seField session)

      -- Update arcs
      cluster = fsTargetCluster shift
      existingArc = filter (\a -> eaCluster a == cluster) (seArcs session)
      newArc = if null existingArc
               then generateArc cluster [shift]
               else head existingArc
      newArcs = newArc : filter (\a -> eaCluster a /= cluster) (seArcs session)
  in session
      { seField = newField
      , seArcs = newArcs
      , seDuration = seDuration session + dt
      , seEvents = seEvents session + 1
      }

-- | Generate session timeline
sessionTimeline :: SessionEpigenetics -> [(Double, GeneCluster, Double)]
sessionTimeline session =
  let arcs = seArcs session
      arcToTimeline arc =
        let time = eaDuration arc * eaProgress arc
            cluster = eaCluster arc
            level = eaStartLevel arc + (eaEndLevel arc - eaStartLevel arc) * eaProgress arc
        in (time, cluster, level)
  in map arcToTimeline arcs

-- =============================================================================
-- Utility Functions
-- =============================================================================

-- | Clamp value to [0, 1]
clamp01 :: Double -> Double
clamp01 x = max 0.0 (min 1.0 x)
