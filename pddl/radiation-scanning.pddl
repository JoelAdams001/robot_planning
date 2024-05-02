(define (domain radiation-scanning)
(:requirements :strips :typing :adl :fluents :durative-actions)

    (:types
        thing region - object
        robot door - thing
        mobile - robot
        quadruped - mobile
    )

    (:predicates
        (has_arm ?rb - robot)
        (located ?rg - region)
        (scanned ?rg - region)

        (robot_location ?rb - robot ?rg - region)
    )

    (:durative-action move
        :parameters (?rb - robot ?rg1 ?rg2 - region)
        :duration ( = ?duration 5)
        :condition (and
            (at start(robot_location ?rb ?rg1))
            (at start(located ?rg2))
            )
        :effect (and
            (at end(not(robot_location ?rb ?rg1)))
            (at end(robot_location ?rb ?rg2))
        )
    )
)